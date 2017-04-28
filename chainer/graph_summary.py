import contextlib
import itertools
import threading
import six
from six.moves import urllib
import sys
import weakref

import numpy
import werkzeug
import werkzeug.serving

from chainer import cuda
from chainer import variable
from chainer import function
from chainer.training import extension
from chainer.training import trainer as trainer_module
from chainer.training import trigger as trigger_module


if cuda.available:
    _ndarrays = (numpy.ndarray, cuda.cupy.ndarray)
else:
    _ndarrays = (numpy.ndarray,)


def get_name(obj):
    if isinstance(obj, variable.VariableNode):
        return obj.name
    elif isinstance(obj, _ndarrays):
        return None
    elif isinstance(obj, function.Function):
        return type(obj).__name__
    elif isinstance(obj, Graph):
        return None
    else:
        assert False


def _get_obj(obj):
    if isinstance(obj, variable.Variable):
        return obj._node
    else:
        return obj


class DataSeriesConfig(object):
    def __init__(self, enable=True, data_reduce=None,
                 preprocess=None, postprocess=None,
                 store_trigger=None, reset_trigger=None):

        if isinstance(data_reduce, DataReduction):
            pass
        elif data_reduce is None or data_reduce == 'overwrite':
            data_reduce = OverwriteReduction()
        elif data_reduce == 'average':
            data_reduce = AverageReduction()
        elif data_reduce == 'mean-std':
            data_reduce = MeanStdReduction()
        elif data_reduce == 'percentile':
            data_reduce = PercentileReduction()
        elif (isinstance(data_reduce, (tuple,list)) and
              len(data_reduce) == 2 and
              callable(data_reduce[0]) and
              callable(data_reduce[1])):
            data_reduce = ReductionByFuncs(
                init_func=data_reduce[0],
                reduce_func=data_reduce[1])
        else:
            raise ValueError("Invalid value for data_reduce.")

        assert isinstance(data_reduce, DataReduction)

        if preprocess is not None:
            assert callable(preprocess)

        if postprocess is not None:
            assert callable(postprocess)

        store_trigger = trigger_module.get_trigger(store_trigger)
        reset_trigger = trigger_module.get_trigger(reset_trigger)

        self.enable = enable
        self.data_reduce = data_reduce
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.store_trigger = store_trigger
        self.reset_trigger = reset_trigger


class GnodeConfig(object):
    def __init__(self, data=None):
        self.data_series_configs = self._make_data_series_configs(data, {})

    def _is_good_config_tuple(self, tup):
        return (len(tup) == 2 and
                isinstance(tup[0], str) and
                isinstance(tup[1], dict))

    def _make_data_series_configs(self, data_spec, configs):
        assert isinstance(configs, dict)

        if isinstance(data_spec, dict):
            data_spec = [('data', data_spec)]

        elif self._is_good_config_tuple(data_spec):
            data_spec = [data_spec]

        if (not isinstance(data_spec, (tuple, list)) or
            not all(len(_) == 2 for _ in data_spec) or
            not all(isinstance(_[0], str) for _ in data_spec) or
            not all(isinstance(_[1], dict) for _ in data_spec)):
            raise ValueError("Invalid data specification.")

        for name, kwargs in data_spec:
            assert isinstance(name, str)
            assert isinstance(kwargs, dict)
            if name not in configs:
                configs[name] = DataSeriesConfig(**kwargs)

        return configs


class DataReduction(object):
    def reset(self):
        pass

    def reduce(self, acc, x, i):
        """Returns new value"""
        pass

    def collect(self, acc, n):
        return acc


class ReductionByFuncs(DataReduction):
    def __init__(self, init_func=None, reduce_func=None):
        assert init_func is not None
        assert reduce_func is not None
        self.init_func = init_func
        self.reduce_func = reduce_func

    def reduce(self, acc, x, i):
        if i == 0:
            return self.init_func(x)
        else:
            return self.reduce_func(acc, x, i)


class OverwriteReduction(DataReduction):
    def reduce(self, acc, x, i):
        return x.copy()


class AverageReduction(DataReduction):
    def reduce(self, acc, x, i):
        if i == 0:
            return x.copy()
        else:
            return acc * (i / float(i+1)) + x / float(i+1)


class MeanStdReduction(DataReduction):
    def __init__(self):
        self._mean = None
        self._mean2 = None

    def reset(self):
        self._mean = None
        self._mean2 = None

    def _std(self):
        return numpy.sqrt(self._mean2 - self._mean * self._mean)

    def reduce(self, acc, x, i):
        if i == 0:
            self._mean = x.copy()
            self._mean2 = x * x
        else:
            self._mean = self._mean * (i / float(i+1)) + x / float(i+1)
            self._mean2 = self._mean2 * (i / float(i+1)) + (x*x) / float(i+1)
        return (self._mean, self._std())


class PercentileReduction(DataReduction):
    def __init__(self, k=10000):
        self._k = k
        self._n_elms = 0

    def reset(self):
        self._n_elms = 0

    def reduce(self, acc, x, i):
        k = self._k
        n_elms = self._n_elms
        reservoir = acc

        if i == 0:
            reservoir = numpy.empty((k,), dtype=x.dtype)

        x_ = x.flat

        # Copy first k elements
        n_copy = max(0, min(x.size, k - n_elms))
        if n_copy > 0:
            reservoir[n_elms:n_elms+n_copy] = x_[0:n_copy]
            n_elms += n_copy

        # Sample remaining elements with probability 1/(n+1)
        #  where n = n_elms, n_elms+1, n_elms+2, ...
        n_sample = x.size - n_copy
        if n_sample > 0:
            j = numpy.random.random((n_sample,)) * numpy.arange(n_elms+1, n_elms+1+n_sample)
            j = j.astype(numpy.int32)
            taken = j < k
            taken_idx = numpy.where(taken)
            reservoir[j[taken_idx]] = x_[taken]
            n_elms += n_sample

        self._n_elms = n_elms

        return reservoir

    def collect(self, acc, n):
        reservoir = acc
        return numpy.percentile(reservoir, numpy.arange(0,110,10))


class DataSeries(object):
    def __init__(self, config):
        self._data_list = []
        self._current_data = None
        self._current_epoch = None
        self._config = config
        self.sample_count = 0
        self._epoch_to_idx = {}

    def get_data(self, index):
        """If index is None, that means the current unfinished data."""

        if index is None:
            # Unfinished data
            epoch = self._current_epoch
            if self._current_data is not None:
                data = self._config.data_reduce.collect(self._current_data, self.sample_count)
                data = self._as_array_recursive(data)
                # Do post-processing on the fly
                if self._config.postprocess:
                    data = self._config.postprocess(data, self.sample_count)
            else:
                data = self._data_list[-1][1]
        else:
            # Finished and stored data
            if not (0 <= index < len(self._data_list)):
                raise IndexError()
            epoch, data = self._data_list[index]
        return epoch, data

    def get_iterations(self):
        epochs = [_[0] for _ in self._data_list]
        # The last `None` represents the latest data
        return epochs + [None]

    def add_sample(self, data, trainer=None):
        assert isinstance(data, _ndarrays)
        config = self._config

        # Preparation
        if config.preprocess:
            data = config.preprocess(data)

        epoch = trainer.updater.epoch_detail

        # Add sample
        assert ((self.sample_count == 0 and self._current_data is None) or
                (self.sample_count > 0 and self._current_data is not None))
        self._current_data = config.data_reduce.reduce(
            self._current_data, data, self.sample_count)
        assert self._current_data is not None
        self._current_epoch = epoch

        self.sample_count += 1

        # Store data
        if config.store_trigger is not None:
            if config.store_trigger(trainer):
                self.store_current(epoch)

        # Reset data
        if config.reset_trigger is not None:
            if config.reset_trigger(trainer):
                self.reset_current()

    def store_current(self, epoch):
        config = self._config
        data = config.data_reduce.collect(self._current_data, self.sample_count)
        data = self._as_array_recursive(data)
        if config.postprocess:
            data = config.postprocess(data, self.sample_count)
        self._data_list.append((epoch, data))

    def reset_current(self):
        self._config.data_reduce.reset()
        self._current_data = None
        self._current_epoch = None
        self.sample_count = 0

    def _as_array_recursive(self, data):
        # TODO: support dict
        if isinstance(data, _ndarrays):
            return self._as_array(data)
        elif isinstance(data, numpy.generic):
            return data
        elif isinstance(data, tuple):
            return tuple([self._as_array_recursive(_) for _ in data])
        elif isinstance(data, list):
            return [self._as_array_recursive(_) for _ in data]
        else:
            assert False, type(data)

    def _as_array(self, data):
        if cuda.available:
            data = cuda.cupy.asnumpy(data)
        assert isinstance(data, numpy.ndarray)
        return data


class DataCollection(object):
    def __init__(self):
        self._data_series_dict = {}

    def __getitem__(self, name):
        return self._data_series_dict[name]

    def __contains__(self, name):
        return name in self._data_series_dict

    def get_names(self):
        return [name for name in self._data_series_dict.keys()
                if self._data_series_dict[name].config.enable]

    def get_summary(self):
        summary = {}
        for name in sorted(self._data_series_dict.keys()):
            data_series = self._data_series_dict[name]
            iter_keys = data_series.get_iterations()
            summary[name] = iter_keys

        return summary


    def add_sample(self, data, trainer=None):
        for name,data_series in self._data_series_dict.items():
            data_series.add_sample(data, trainer)

    def ensure_data_series_prepared(self, name, config):
        assert isinstance(name, str)
        assert isinstance(config, DataSeriesConfig)
        if name not in self._data_series_dict:
            self._data_series_dict[name] = DataSeries(config)


class Gnode(object):
    def __init__(self, obj_type, tag, metatag, name, extra_clue):
        assert tag is None or isinstance(tag, str)
        assert isinstance(obj_type, type)

        self.tag = tag
        self.metatag = metatag
        self.obj_type = obj_type
        self.in_edges = set()
        self.out_edges = set()
        self.extra_clue = extra_clue
        self.kind = Gnode._get_kind(obj_type)

        # node_config could either be dict or GnodeConfig.
        self.node_config = None

        # Variable: .name
        # Function: .name
        # Graph: None
        self.name = name

    def set_node_config(self, node_config):
        assert isinstance(node_config, (GnodeConfig, dict))
        if isinstance(node_config, dict):
            assert 'data' not in node_config
        self.node_config = node_config

    @property
    def clue(self):
        return (self.kind, self.tag, self.metatag, self.name) + self.extra_clue

    def __repr__(self):
        name = 'Gnode' if self.tag is None else 'Gnode@{}'.format(self.tag)
        return '<{} {:x} in={} out={} type={}>'.format(
            name, id(self), len(self.in_edges), len(self.out_edges), self.obj_type.__name__)

    @classmethod
    def _get_kind(cls, obj_type):
        if obj_type is variable.VariableNode:
            return 'variable'
        if obj_type in _ndarrays:
            return 'variable'
        elif obj_type is Graph:
            return 'subgraph'
        elif issubclass(obj_type, function.Function):
            return 'function'
        else:
            assert False

    @classmethod
    def from_obj(cls, obj, tag, metatag):
        assert not isinstance(obj, Graph)
        if isinstance(obj, (variable.VariableNode,) + _ndarrays):
            return VariableGnode(obj, tag, metatag)
        elif isinstance(obj, function.Function):
            return FunctionGnode(obj, tag, metatag)

        clue = cls.get_extra_clue(obj, tag)
        return Gnode(
            type(obj), tag, get_name(obj), clue)

    @classmethod
    def get_clue(cls, obj, tag, metatag):
        kind = cls._get_kind(type(obj))
        return (kind, tag, metatag, get_name(obj)) + cls.get_extra_clue(obj, tag)

    @classmethod
    def get_extra_clue(cls, obj, tag):
        if isinstance(obj, (variable.VariableNode,) + _ndarrays):
            extra_clue = VariableGnode.get_extra_clue(obj, tag)
        elif isinstance(obj, function.Function):
            extra_clue = FunctionGnode.get_extra_clue(obj, tag)
        elif isinstance(obj, Graph):
            extra_clue = cls.get_graph_clue(obj, tag)
        else:
            assert False
        return extra_clue

    def get_compatible_in_gnodes(self, obj, tag, metatag, arg_index):
        if len(self.in_edges) > 0:
            clue = Gnode.get_clue(obj, tag, metatag)
            in_gnodes = []
            for edge in self.in_edges:
                if edge.arg_index != arg_index:
                    continue
                if clue == edge.in_gnode.clue:
                    in_gnodes.append(edge.in_gnode)
            return in_gnodes
        else:
            return []

    def get_out_edge(self, out_gnode):
        for edge in self.out_edges:
            if edge.out_gnode == out_gnode:
                return edge
        return None

    @classmethod
    def from_graph(cls, obj, tag, metatag):
        return Gnode(
            type(obj), tag, metatag, get_name(obj),
            cls.get_graph_clue(obj, tag))

    @classmethod
    def get_graph_clue(cls, graph, tag):
        return (graph.tag,)


class VariableGnode(Gnode):
    def __init__(self, var, tag, metatag):
        assert isinstance(var, (variable.VariableNode,) + _ndarrays)
        name = var.name if isinstance(var, variable.VariableNode) else None
        extra_clue = VariableGnode.get_extra_clue(var, tag)
        super(VariableGnode, self).__init__(type(var), tag, metatag, name, extra_clue)

        self.shape = var.shape
        self.dtype = var.dtype
        self.name = name

        self.data_collection = DataCollection()

    @classmethod
    def get_extra_clue(cls, var, tag):
        assert isinstance(var, (variable.VariableNode,) + _ndarrays)
        return (var.shape, var.dtype, var.name if isinstance(var, variable.VariableNode) else None)

    def __repr__(self):
        name = self.__class__.__name__
        if self.tag is not None:
            name += '@' + self.tag
        lst = [
            name,
            ('\'' + self.name + '\'') if self.name else None,
            '{:x}'.format(id(self)), self.shape, self.dtype,
            'fp={}'.format(self.clue),
        ]
        return '<{}>'.format(
            ' '.join(str(_) for _ in lst if _ is not None))

    def _prepare_data_collection(self, data_series_configs):
        assert isinstance(data_series_configs, dict)
        for name, config in data_series_configs.items():
            if config.enable:
                self.data_collection.ensure_data_series_prepared(name, config)

    def add_data_sample(self, data, trainer):
        assert isinstance(data, _ndarrays)
        node_config = self.node_config
        if node_config is None:
            return

        self._prepare_data_collection(node_config.data_series_configs)
        self.data_collection.add_sample(data, trainer)


class FunctionGnode(Gnode):
    def __init__(self, func, tag, metatag):
        extra_clue = FunctionGnode.get_extra_clue(func, tag)
        super(FunctionGnode, self).__init__(type(func), tag, metatag, type(func).__name__, extra_clue)

    @classmethod
    def get_extra_clue(cls, obj, tag):
        assert isinstance(obj, function.Function)
        inputs = obj.inputs
        outputs = [_() for _ in obj.outputs]
        assert all(_ is not None for _ in inputs)
        assert all(_ is not None for _ in outputs)
        assert all(isinstance(_, variable.VariableNode) for _ in inputs)
        assert all(isinstance(_, variable.VariableNode) for _ in outputs)
        return (
            tuple([(_.shape, _.dtype, _.name) for _ in inputs]),
            tuple([(_.shape, _.dtype, _.name) for _ in outputs]),
        )

    def get_out_edges(self, arg_index):
        return [edge for edge in self.out_edges if edge.arg_index == arg_index]

    def get_in_edges(self, arg_index):
        return [edge for edge in self.in_edges if edge.arg_index == arg_index]


class GraphEdge(object):
    def __init__(self, in_gnode, out_gnode, arg_index):
        """
        arg_index: The index of the argument to which the edge is connected to/from.
        """
        assert in_gnode is None or isinstance(in_gnode, Gnode), type(in_gnode)
        assert isinstance(out_gnode, Gnode)
        assert isinstance(arg_index, int)
        self.in_gnode = in_gnode
        self.out_gnode = out_gnode
        self.hash_key = (in_gnode, out_gnode, arg_index)

        self.count = 0
        self.arg_index = arg_index

        self.data_sum = None

    def __hash__(self):
        return hash(self.hash_key)

    def __eq__(self, other):
        return self.hash_key == other.hash_key

    def __repr__(self):
        return '<GraphEdge {:x} i={} from={} to={}>'.format(
            id(self), self.arg_index, self.in_gnode, self.out_gnode)


class Graph(object):
    def __init__(self, tag, inherited_node_configs=None):
        self.tag = tag
        self.nodes = set()
        self.node_map = {} # tag -> node
        self.subgraphs = {}
        self.subgraph_inout_gnodes = {} # tag -> (input nodes, output nodes)

        self.subgraph_output_tag_map = {} # output variable tag -> (subgraph tag, arg_index)

        self.input_nodes = None
        self.output_nodes = None
        self._lock = threading.Lock()

        self._inherited_node_configs = inherited_node_configs
        self._node_configs = None

    def lock(self):
        self._lock.acquire()

    def unlock(self):
        self._lock.release()

    def config_node(self, tag_path, **kwargs):
        tag_path = tag_path.split('/')

        # Make hierarchical config
        d = self._node_configs
        if d is None:
            d = self._node_configs = {}

        for tag in tag_path[:-1]:
            subd = d.get(tag)
            if subd is None:
                subd = d[tag] = {}
            d = subd

        d[tag_path[-1]] = GnodeConfig(**kwargs)

    def get_node_config(self, tag):
        """Returns a GnodeConfig (leaf) or a dict (subgraph) or None (not found)"""

        if self._node_configs is not None:
            config = self._node_configs.get(tag)
            if config is not None:
                return config

        if self._inherited_node_configs is not None:
            config = self._inherited_node_configs.get(tag)
            if config is not None:
                return config

        return None

    @property
    def is_empty(self):
        return len(self.nodes) == 0

    @property
    def edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.in_edges)
            edges.update(node.out_edges)
        return edges

    def set_tag(self, node, tag):
        assert tag not in self.node_map
        assert node in self.nodes
        self.node_map[tag] = node

    def find_node(self, tag):
        assert isinstance(tag, str)
        return self.node_map.get(tag)

    def submit_outputs(self, output_variables):
        if self.output_nodes is None:
            self.output_nodes = []
            for i,(obj, tag, metatag) in enumerate(output_variables):
                obj_ = _get_obj(obj)
                gnode = Gnode.from_obj(obj_, tag, metatag)
                self._add_node(gnode)

                self.output_nodes.append(gnode)

        assert len(self.output_nodes) == len(output_variables)

    def submit_inputs(self, input_tuples):
        if self.input_nodes is None:
            self.input_nodes = []
            for i,(obj,tag,metatag) in enumerate(input_tuples):
                assert tag is None or self.find_node(tag) is None
                obj_ = _get_obj(obj)
                gnode = Gnode.from_obj(obj_, tag, metatag)
                self._add_node(gnode)
                self.input_nodes.append(gnode)

        assert len(self.input_nodes) == len(input_tuples)

    def get_node(self, tag):
        return self.node_map.get(tag)

    def get_subgraph(self, tag, create=False):
        assert isinstance(tag, str)
        subgraph = self.subgraphs.get(tag)
        if subgraph is None:
            if not create:
                return None, None

            subconfigs = self.get_node_config(tag)

            subgraph = Graph(tag, subconfigs)
            self.subgraphs[tag] = subgraph
            node = self._add_graph_node(subgraph, tag, None)
        else:
            node = self.node_map[tag]
        return subgraph, node

    def get_compatible_nodes(self, obj, tag, metatag):
        clue = Gnode.get_clue(obj, tag, metatag)
        nodes = []
        for node in self.nodes:
            if node.clue == clue:
                nodes.append(node)
        return nodes

    def _add_graph_node(self, obj, tag, metatag):
        assert isinstance(obj, Graph)
        assert isinstance(tag, str)
        node = self.node_map.get(tag, None)
        if node is None:
            node = Gnode.from_graph(obj, tag, metatag)
            self._add_node(node)
        return node

    def _add_node(self, node):
        if node not in self.nodes:
            if node.tag is not None:
                if node.tag in self.node_map:
                    raise RuntimeError('Duplicate tag is detected.')

                # Store node config if any
                node_config = self.get_node_config(node.tag)
                if node_config is not None:
                    node.set_node_config(node_config)

            self.node_map[node.tag] = node
            self.nodes.add(node)
            self.debug("{}: Gnode added: {}".format(self.tag, node))

    def add_node(self, obj, tag):
        gnode = self.node_map.get(tag)
        if gnode is None:
            gnode = Gnode.from_obj(obj, tag)
            self._add_node(gnode)
        return gnode

    def assert_consistent(self):
        # Test
        for tag,node in self.node_map.items():
            assert node in self.nodes

        for node in self.nodes:
            if node.tag is not None:
                assert self.node_map[node.tag] == node

        # Test that node tags do not conflict among nodes
        node_tags = [node.tag for node in self.nodes if node.tag is not None]
        assert len(node_tags) == len(set(node_tags))

        # No node should be isolated
        for node in self.nodes:
            assert len(node.in_edges) > 0 or len(node.out_edges) > 0

        # Test node connections
        for node in self.nodes:
            for in_edge in node.in_edges:
                assert in_edge.out_gnode == node
                in_gnode = in_edge.in_gnode
                if in_gnode is not None:
                    assert in_gnode in self.nodes
                    assert any(_.out_gnode == node for _ in in_gnode.out_edges)

            for out_edge in node.out_edges:
                assert out_edge.in_gnode == node

                out_gnode = out_edge.out_gnode
                if out_gnode is not None:
                    assert out_gnode in self.nodes
                    assert any(_.in_gnode == node for _ in out_gnode.in_edges)

    def debug(self, s):
        pass

    def debug_dump(self, out=None):
        if out is None:
            out = sys.stderr

        def putline(s):
            out.write(s)
            out.write('\n')

        def debug(s):
            self.debug(s)

        debug("---")
        putline('digraph graphname {rankdir=TB;')
        visited_nodes = set()
        written_nodes = set()

        nodes = set([(_, 0) for _ in self.nodes if len(_.out_edges) == 0])
        while len(nodes) > 0:
            node, depth = nodes.pop()
            print_prefix = "Dump {}".format("  " * depth)
            debug("{}{}".format(print_prefix, node))
            if id(node) in visited_nodes:
                continue
            visited_nodes.add(id(node))


            if id(node) not in written_nodes:
                if node.obj_type is variable.VariableNode:
                    shape = 'oval'
                    label = str(node)
                elif node.obj_type in _ndarrays:
                    shape = 'oval'
                    label = 'ndarray'
                elif issubclass(node.obj_type, function.Function):
                    shape = 'box'
                    label = "{}\\n{}".format(node, node.clue)
                elif node.obj_type is Graph:
                    shape = 'doubleoctagon'
                    label = str(node)
                else:
                    assert False
                putline('{} [label="{}", shape="{}"];'.format(
                    id(node),
                    label, shape))
                written_nodes.add(id(node))

            debug("{}In edges: {}".format(print_prefix, node.in_edges))

            for in_edge in sorted(node.in_edges, key=lambda edge: edge.arg_index):
                debug("{}In edge: {}".format(print_prefix, in_edge))
                debug("{}In edge@: {}".format(print_prefix, in_edge.in_gnode.out_edges))
                in_gnode = in_edge.in_gnode
                assert in_gnode is not None

                if id(in_gnode) not in visited_nodes:
                    nodes.add((in_gnode, depth+1))

                putline('{} -> {} [label="i={} n={}"];'.format(
                    id(in_gnode), id(node), in_edge.arg_index, in_edge.count))

        putline('}')


class MatchConfig:
    def __init__(self, can_create_edge=False, can_create_node=False, max_create=0):
        self.can_create_edge = can_create_edge
        self.can_create_node = can_create_node
        self.max_create = max_create


class MatchNode:
    def __init__(self, obj, gnode, prev_mnode, arg_index):
        assert gnode is None or isinstance(gnode, Gnode)
        assert prev_mnode is None or isinstance(prev_mnode, MatchNode)
        self.obj = obj
        self.gnode = gnode
        self.prev_mnode = prev_mnode
        self.arg_index = arg_index

        self.submitted = False

    @property
    def out_edge(self):
        if self.prev_mnode is None:
            return None
        if self.gnode is None:
            return None
        prev_gnode = self.prev_mnode.gnode
        out_edge = self.gnode.get_out_edge(prev_gnode)
        return out_edge


class MatchState:
    def __init__(self, prev_state, gnode, arg_index, create_edge, create_node, obj):
        self.prev_state = prev_state
        self.gnode = gnode
        self.arg_index = arg_index
        if prev_state is None:
            self.depth = 0
            self.created_edges = 0
            self.created_nodes = 0
            self.visited_objs = {}
        else:
            assert prev_state.good
            self.depth = prev_state.depth + 1
            self.created_edges = prev_state.created_edges
            self.created_nodes = prev_state.created_nodes
            self.visited_objs = prev_state.visited_objs

        if create_edge:
            self.created_edges += 1
        if create_node:
            self.created_nodes += 1

        self.obj = obj
        self.good = True

    def __enter__(self):
        if not isinstance(self.obj, Graph):
            if id(self.obj) in self.visited_objs:
                self.good = False

        if self.good:
            self.visited_objs[id(self.obj)] = self.visited_objs.get(id(self.obj), 0) + 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.good:
            if self.visited_objs[id(self.obj)] == 1:
                del self.visited_objs[id(self.obj)]
            else:
                self.visited_objs[id(self.obj)] -= 1
        self.good = True

    @property
    def prev_mnode(self):
        if self.prev_state is None:
            return None
        else:
            return self.prev_state.mnode


class MatchSolution:
    def __init__(self, input_nodes=None, floating_nodes=None):
        self.input_nodes = input_nodes or []
        self.floating_nodes = floating_nodes or []

    def merge(self, other):
        assert isinstance(other, MatchSolution)
        self.input_nodes += other.input_nodes
        self.floating_nodes += other.floating_nodes

    def empty(self):
        return len(self.input_nodes) == 0 and len(self.floating_nodes) == 0

    def leaf_nodes(self):
        return self.input_nodes + self.floating_nodes


class GraphContext(object):
    def __init__(self, tag, graph):
        self.tag = tag
        self.graph = graph
        self._closed = False

        self._output_variables_of_last_pass = []

    def start_pass(self, inputs, trainer=None):
        # See: comment of end_pass()
        last_outputs = [_() for _ in self._output_variables_of_last_pass]
        last_outputs = [_ for _ in last_outputs if _ is not None]
        self._init_pass(inputs, last_outputs, trainer)

    def end_pass(self):
        # Weakly keep the output variables of this pass.
        # They are used to identify the input variables of the next pass, if omitted.
        # If a weak reference is invalidated, that means the variable is no longer used
        # and cannot be a part of the input variables.
        self._output_variables_of_last_pass = [
            weakref.ref(v) for v,_,_ in self.output_variables]

        self._cleanup_pass()

    def _init_pass(self, inputs, last_outputs, trainer):
        self._closed = False
        self.nodes = set()
        self._var_unnamed_tag_counter = 0
        self.buffered_node_configs = None
        self.trainer = trainer

        # chainer.Variable
        self.output_variables = None
        self.input_variables = inputs
        self.last_output_variables = last_outputs

        self.input_variable_set = set([id(_get_obj(_)) for _ in inputs])

        # chainer.VariableNode -> tag, metatag
        input_metatags = ['$i{}'.format(i) for i in range(len(inputs))]
        self.variable_map = {}
        for i,v in enumerate(inputs):
            obj_ = _get_obj(v)
            self.variable_map[id(obj_)] = (None, input_metatags[i])

        # tag -> chainer.VariableNode
        self.variable_map2 = {}

        # chainer.VariableNode -> chainer.Variable
        self.variable_node_map = {}

        self.subgraph_map = {} # subgraph tag -> (input variables, output variables)
        self.subgraph_output_map = {} # output variable -> (subgraph, index, gnode)

        # Submit input variables to the graph
        for v in inputs:
            if isinstance(v, variable.Variable):
                self._memorize_variable(v)

    def _cleanup_pass(self):
        # TODO: These variables and related operations could be capsulated into
        #       a separate class.
        del self.nodes
        del self.buffered_node_configs
        del self.output_variables
        del self.input_variables
        del self.input_variable_set
        del self._var_unnamed_tag_counter
        del self.variable_map
        del self.variable_map2
        del self.variable_node_map
        del self.subgraph_map
        del self.subgraph_output_map
        del self.trainer

    def set_output(self, outputs):
        self.debug("set_output: {}".format(outputs))
        assert not self._closed
        output_variables = []

        output_metatags = ['$o{}'.format(i) for i in range(len(outputs))]

        for var,metatag in zip(outputs, output_metatags):
            obj_ = _get_obj(var)
            tag, metatag_ = self._get_variable_tags(var)
            assert metatag_ is None

            self.variable_map[id(obj_)] = (tag, metatag)
            output_variables.append((var, tag, metatag))

        self.output_variables = output_variables
        self._closed = True

        #
        self.debug("set_output: {}".format(self.subgraph_output_map))
        self.debug("set_output: {}".format(self.subgraph_output_map))
        # Create output variables to the graph
        self.graph.submit_outputs(self.output_variables)

    def _memorize_variable(self, obj):
        """
        Memorize mapping from variable.VariableNode to variable.Variable
        and prevent automatic data release
        """
        assert isinstance(obj, variable.Variable)
        obj_ = _get_obj(obj)
        self.variable_node_map[id(obj_)] = obj

    def _is_input_variable(self, obj):
        if id(obj) in self.input_variable_set:
            return True
        if isinstance(obj, variable.VariableNode):
            return id(obj.data) in self.input_variable_set
        return False

    def _get_variable_tags(self, obj):
        obj_ = _get_obj(obj)
        tup = self.variable_map.get(id(obj_), None)

        # xp.ndarray will be converted to Variable (and VariableNode) by a Function.
        # Original xp.ndarray can be found in VariableNode.data.
        if tup is None and isinstance(obj, variable.VariableNode):
            obj_ = obj.data
            tup = self.variable_map.get(id(obj_), None)

        if tup is None:
            return None, None
        else:
            assert len(tup) == 2
            return tup

    def set_tag(self, obj, tag):
        assert isinstance(obj, (variable.Variable,) + _ndarrays)
        assert isinstance(tag, str)
        assert tag not in self.variable_map2
        if isinstance(obj, variable.Variable):
            self._memorize_variable(obj)

        tag_, metatag_ = self._get_variable_tags(obj)
        if tag_ is not None:
            raise RuntimeError("Variable is already tagged as '{}'".format(
                tag_))

        obj_ = _get_obj(obj)
        self.variable_map[id(obj_)] = (tag, metatag_)
        self.variable_map2[tag] = obj_
        self.debug("set_tag({})".format(tag))
        self.debug("{}, {}".format(id(self), self.variable_map2.keys()))

    def _ensure_variable_has_tag(self, obj):
        tag, metatag = self._get_variable_tags(obj)
        if tag is None:
            tag = 'var{}'.format(self._var_unnamed_tag_counter)
            self._var_unnamed_tag_counter += 1
            self.set_tag(obj, tag)

        return tag

    def config_node(self, target, **kwargs):
        # If target is tag (=str), just store the tag for future statistics collection.
        # If it's a variable (ndarray or Variable), attach a tag (if not yet) to it
        # and use that tag.

        if isinstance(target, str):
            tag = target
        elif isinstance(target, (variable.Variable,) + _ndarrays):
            tag = self._ensure_variable_has_tag(target)
        else:
            assert False

        #
        self.debug("config_node: {}".format(tag))
        if self.buffered_node_configs is None:
            self.buffered_node_configs = []
        self.buffered_node_configs.append((
            tag, GnodeConfig(**kwargs)
        ))

    def debug(self, s):
        pass #print("{}: {}".format(self.tag, s))

    def _find_matches_partial_tree(self, obj, prev_mnode, state, config):
        """ Returns: [] of MatchSolution
        """

        rec = self._find_matches_partial_tree

        depth = state.depth
        gnode = state.gnode
        arg_index = state.arg_index

        assert isinstance(gnode, Gnode)
        assert prev_mnode is None or isinstance(prev_mnode, MatchNode)

        def debug(s):
            self.debug("{}{}".format("   " * depth, s))

        if state.created_edges + state.created_nodes > config.max_create:
            debug("Exceed: {} + {} > {}".format(
                state.created_edges, state.created_nodes, config.max_create))
            return [], True

        debug("Obj = {!r}".format(obj))
        debug("gnode = {}".format(gnode))


        if self._is_input_variable(obj):
            mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
            debug("Reached input variable: {}".format(id(obj)))
            return [MatchSolution([mnode], [])], False

        else:
            mnode = MatchNode(obj, gnode, prev_mnode, arg_index)

            if isinstance(obj, variable.VariableNode):
                func = obj.creator
                if func is None:
                    # This is an automatically-created variable.
                    mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
                    debug("Reached floating variable: {}".format(obj))
                    return [MatchSolution([], [mnode])], False

                subgraph_tuple = self.subgraph_output_map.get(id(obj))
                if subgraph_tuple is not None:
                    #subgraph_tag, index = subgraph_tuple
                    subgraph, index = subgraph_tuple
                    # This is an output variable of some subgraph
                    inputs = [subgraph]
                    arg_indices = [index]

                else:
                    # Ordinary variable
                    inputs = [func]
                    arg_indices = [i for i,_ in enumerate(func.outputs) if _() is obj]

            elif isinstance(obj, _ndarrays):
                # This is an automatically-created variable.
                mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
                debug("Reached floating variable: {}".format(id(obj)))
                return [MatchSolution([], [mnode])], False

            elif isinstance(obj, function.Function):
                inputs = obj.inputs
                debug("FUNC: {}".format(type(obj)))
                debug("  : {}".format(' '.join('{}'.format(type(_)) for _ in inputs)))
                arg_indices = list(range(len(inputs)))

            elif isinstance(obj, Graph):
                assert obj.input_nodes is not None
                assert obj.output_nodes is not None
                subgraph_tag = obj.tag
                assert isinstance(subgraph_tag, str)

                inputs, _ = self.subgraph_map[subgraph_tag]
                _ = None

                arg_indices = list(range(len(inputs)))
            else:
                assert False

            assert all(isinstance(_, (variable.VariableNode, function.Function, Graph) + _ndarrays) for _ in inputs)
            assert len(arg_indices) == len(inputs)


            # Find solutions for each input argument
            in_arg_solutions = [[] for _ in range(len(inputs))]
            hit_creation_limit = False

            for in_obj, in_arg_index in zip(inputs, arg_indices):
                if isinstance(in_obj, (variable.VariableNode,) + _ndarrays):
                    in_tag, in_metatag = self._get_variable_tags(in_obj)
                elif isinstance(in_obj, Graph):
                    in_tag = in_obj.tag
                    in_metatag = None
                else:
                    in_tag = None
                    in_metatag = None

                debug('FIND MATCH: {} {} {}'.format(in_tag, in_metatag, in_arg_index))

                # (1) Examine known input gnodes
                for in_gnode in gnode.get_compatible_in_gnodes(in_obj, in_tag, in_metatag, in_arg_index):
                    with MatchState(state, in_gnode, in_arg_index, False, False, in_obj) as st:
                        if st.good:
                            sols, hit_cl = rec(in_obj, mnode, st, config)
                            in_arg_solutions[in_arg_index] += sols
                            hit_creation_limit = hit_creation_limit or hit_cl

                if len(in_arg_solutions[in_arg_index]) == 0:
                    # (2) Examine compatible gnodes
                    if config.can_create_edge and not (in_tag is None and in_metatag is None):
                        debug('(2)')
                        for in_gnode in self.graph.get_compatible_nodes(in_obj, in_tag, in_metatag):
                            with MatchState(state, in_gnode, in_arg_index, True, False, in_obj) as st:
                                if st.good:
                                    sols, hit_cl = rec(in_obj, mnode, st, config)
                                    in_arg_solutions[in_arg_index] += sols
                                    hit_creation_limit = hit_creation_limit or hit_cl

                    # (3) Examine new gnode
                    to_create_node = (
                        config.can_create_node and
                        (not isinstance(in_obj, Graph)))

                    if to_create_node:
                        in_gnode = Gnode.from_obj(in_obj, in_tag, in_metatag)
                        debug("create: {} {}".format(in_tag, in_obj))
                        debug("create2: {}".format(self.variable_map))
                        debug("create3: {}".format(in_gnode.clue))
                        with MatchState(state, in_gnode, in_arg_index, True, True, in_obj) as st:
                            if st.good:
                                sols, hit_cl = rec(in_obj, mnode, st, config)
                                in_arg_solutions[in_arg_index] += sols
                                hit_creation_limit = hit_creation_limit or hit_cl

                    # If solution cannnot be found for any of input arguments,
                    # this node all in all is a failure.
                    if len(in_arg_solutions[in_arg_index]) == 0:
                        debug("No solution")
                        return [], hit_creation_limit

            # Solution(s) are found
            solutions = []
            for prod in itertools.product(*in_arg_solutions):
                i_sol_nodes = sum((prod[_].input_nodes for _ in range(len(prod))), [])
                f_sol_nodes = sum((prod[_].floating_nodes for _ in range(len(prod))), [])
                if len(i_sol_nodes) != len(set(i_sol_nodes)):
                    continue
                if len(f_sol_nodes) != len(set(f_sol_nodes)):
                    continue
                solutions.append(MatchSolution(i_sol_nodes, f_sol_nodes))

            return solutions, hit_creation_limit

    def submit_subgraph(self, subgraph, input_variables, output_variables):
        self.debug("submit_subgraph: {}".format(subgraph.tag))

        subgraph_tag = subgraph.tag
        assert isinstance(subgraph_tag, str)
        assert subgraph_tag not in self.subgraph_map
        input_objs = [_get_obj(_) for _ in input_variables]
        self.subgraph_map[subgraph_tag] = (input_objs, output_variables)

        for i,out_var in enumerate(output_variables):
            self.subgraph_output_map[id(_get_obj(out_var))] = (subgraph, i)

    def submit_graph(self):
        # Submit input variables to the graph
        input_tuples = []
        for v in self.input_variables:
            obj_ = _get_obj(v)
            tag, metatag = self.variable_map[id(obj_)]
            input_tuples.append((v, tag, metatag))

        self.graph.submit_inputs(input_tuples)

        # Find the best match
        solution = self._find_best_match_tree()
        assert not solution.empty()
        assert len(solution.input_nodes) == len(self.input_variables)

        # Create graph nodes and edges according to the match tree
        mnodes = self._submit_match_tree(solution)

        # Submit buffered node configs
        self._submit_buffered_node_configs()

        # Submit data statistics
        self._submit_data_statistics(mnodes)

        # Debug
        self.graph.assert_consistent()

    def _find_best_match_tree(self):
        if self.graph.is_empty:
            configs = (MatchConfig(can_create_edge=True, can_create_node=True, max_create=_) for _ in itertools.count(1))
        else:
            configs = itertools.chain(
                [
                    MatchConfig(can_create_edge=False, can_create_node=False, max_create=0),
                ],
                (MatchConfig(can_create_edge=True, can_create_node=True, max_create=_) for _ in itertools.count(1)))

        for config in configs:
            all_output_solved = True
            hit_creation_limit = False
            solution = MatchSolution()
            for (out_var, _, _), out_gnode in zip(self.output_variables, self.graph.output_nodes):
                with MatchState(None, out_gnode, None, False, False, out_var) as st:
                    sols, hit_cl = self._find_matches_partial_tree(out_var._node, None, st, config)
                    hit_creation_limit = hit_creation_limit or hit_cl
                if len(sols) == 0:
                    all_output_solved = False
                    break

                # Takes the first solution
                solution.merge(sols[0])

            if all_output_solved:
                break

            if config.max_create > 0 and not hit_creation_limit:
                # Solution has not been found, and node/edge creation limit
                # has not been reached. That means there's some glitch in
                # matching logic...
                assert False

        self.debug("Solution found")

        return solution

    def _submit_match_tree(self, match_solution):
        # Traverses the tree from input nodes

        front = set(match_solution.leaf_nodes())
        submitted_mnodes = set()

        while len(front) > 0:
            mnode = front.pop()
            if mnode.submitted:
                continue

            mnode.submitted = True
            submitted_mnodes.add(mnode)

            gnode = mnode.gnode
            self.graph._add_node(gnode)

            prev_mnode = mnode.prev_mnode
            if prev_mnode is not None:
                prev_gnode = prev_mnode.gnode

                out_edge = gnode.get_out_edge(prev_gnode)
                if out_edge is None:
                    out_edge = GraphEdge(gnode, prev_gnode, mnode.arg_index)
                    prev_gnode.in_edges.add(out_edge)
                    gnode.out_edges.add(out_edge)

                # Edge statistics
                out_edge.count += 1

                front.add(prev_mnode)

        return submitted_mnodes

    def _submit_buffered_node_configs(self):
        if self.buffered_node_configs is None:
            return
        for tag, node_config in self.buffered_node_configs:
            gnode = self.graph.node_map[tag]
            if not isinstance(gnode, VariableGnode):
                raise NotImplementedError('Currently only variable nodes have data statistics')
            gnode.set_node_config(node_config)

        self.buffered_node_configs = None

    def _submit_data_statistics(self, mnodes):
        # Node statistics
        for mnode in mnodes:
            gnode = mnode.gnode
            out_edge = mnode.out_edge

            if isinstance(mnode.obj, (variable.VariableNode,) + _ndarrays):
                if gnode.node_config is not None:
                    if isinstance(mnode.obj, variable.VariableNode):
                        data = self.variable_node_map[id(mnode.obj)].data
                    else:
                        data = mnode.obj
                    self.debug('Add data sample: {}'.format(gnode.tag))
                    gnode.add_data_sample(data, self.trainer)


class DummyGraphContext(object):
    def __init__(self):
        pass

    def set_output(self, outputs):
        pass

    def config_node(self, *args, **kwargs):
        pass

    def set_tag(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def root_graph(input_variables, graph, trainer=None, context=None):
    assert trainer is None or isinstance(trainer, trainer_module.Trainer)
    assert context is None or isinstance(context, GraphContext)
    current_thread = threading.current_thread()

    graph.lock()

    if context is None:
        context = GraphContext(graph.tag, graph)
    current_thread.__dict__['graph_context'] = context
    context.start_pass(input_variables, trainer=trainer)

    try:
        yield context
    finally:
        context.submit_graph()
        context.end_pass()
        current_thread.__dict__['graph_context'] = None
        graph.unlock()


class graph(object):
    def __init__(self, input_variables, tag, enable=True):
        assert isinstance(tag, str)
        self.input_variables = input_variables
        self.tag = tag
        self.enable = enable

    def cleanup(self):
        self.input_variables = None
        self.graph = None

    def __enter__(self):
        if not self.enable:
            context = DummyGraphContext()
        else:
            current_thread = threading.current_thread()
            outer_context = current_thread.__dict__.get('graph_context', None)

            input_variables = self.input_variables
            tag = self.tag

            if outer_context is None:
                # There's no root graph.
                self.enable = False
                return DummyGraphContext()

            # Take the graph from the outer context
            graph, _ = outer_context.graph.get_subgraph(tag, create=True)

            context = GraphContext(tag, graph)
            current_thread.__dict__['graph_context'] = context

            context.start_pass(input_variables, trainer=outer_context.trainer)

            self.outer_context = outer_context

        self.context = context
        return context

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable:
            outer_context = self.outer_context

            if exc_type is None:
                context = self.context
                input_variables = self.input_variables
                graph = context.graph
                context.submit_graph()

                outer_context.submit_subgraph(
                    graph,
                    input_variables,
                    [v for v,_,_ in context.output_variables])

                if not context._closed:
                    raise RuntimeError(
                        "You must call GraphContext.set_output()"
                        " at the end of the graph context.")

                context.end_pass()

            current_thread = threading.current_thread()
            current_thread.__dict__['graph_context'] = outer_context

        self.cleanup()


class GraphSummary(extension.Extension):
    trigger = lambda a,b: True
    invoke_before_update = True
    invoke_after_update = True

    def __init__(self, graph, keys):
        self.graph = graph
        self.keys = keys
        self._ctx = None
        self.context = GraphContext(graph.tag, graph)

    def __call__(self, trainer):
        if self._ctx is None:
            # Before update
            self._ctx = root_graph([], self.graph, trainer, self.context)
            self._ctx.__enter__()
        else:
            # After update
            outputs = [trainer.observation[_] for _ in self.keys]
            self.context.set_output(outputs)
            self._ctx.__exit__(None, None, None)
            self._ctx = None


server_graph = None


class ErrorResponse(Exception):
    pass


def _do_run_server():
    werkzeug.serving.run_simple('localhost', 6007, graph_app)

def run_server(graph, async=False):
    global server_graph
    server_graph = graph

    if async:
        t = threading.Thread(target=_do_run_server)
        t.start()
    else:
        _do_run_server()

import json

def get_obj(path):
    """
    Returns one of (Graph, Gnode)
    """

    if len(path) == 0:
        path = server_graph.tag

    path_list = path.split('/')

    graph_name = path_list.pop(0)
    graph = server_graph
    if graph_name != graph.tag:
        raise RuntimeError()

    while len(path_list) > 0:
        tag = path_list.pop(0)

        # Subgraph?
        graph_, _ = graph.get_subgraph(tag)
        if graph_ is not None:
            graph = graph_
            continue

        # Node?
        node = graph.get_node(tag)
        if node is not None:
            if len(path_list) > 0:
                raise KeyError("Invalid object path: {}".format(path))
            return node

        raise KeyError("Object not found: {}".format(path))

    return graph


def _ndarray_to_list(data):
    if isinstance(data, numpy.ndarray):
        return data.tolist()
    if isinstance(data, numpy.generic):
        return data.item()
    elif isinstance(data, (tuple,list)):
        return [_ndarray_to_list(_) for _ in data]
    elif isinstance(data, dict):
        return {key: _ndarray_to_list(_) for key,_ in data.items()}
    else:
        assert data is None or isinstance(data, (float,str) + six.integer_types), type(data)
        return data


def api(api_name, path, query, environ):
    method = environ['REQUEST_METHOD']
    if api_name == 'graph' and method == 'GET':
        nodes = []
        edges = []

        if len(path) == 0:
            path = server_graph.tag

        graph = get_obj(path)
        if not isinstance(graph, Graph):
            raise KeyError(
                'No such graph: {}\n'.format(path))

        graph.lock()
        try:
            # TODO: Should not return object id

            for node in graph.nodes:
                d_node = {
                    'id': id(node),
                }

                # type
                if node.obj_type in (variable.VariableNode,) + _ndarrays:
                    d_node['type'] = 'variable'
                    d_node['shape'] = list(node.shape)
                    d_node['dtype'] = node.dtype.name
                elif node.obj_type is Graph:
                    d_node['type'] = 'subgraph'
                    d_node['path'] = '{}/{}'.format(path, node.tag)
                elif issubclass(node.obj_type, function.Function):
                    d_node['type'] = 'function'
                else:
                    assert False

                # name
                if node.name is not None:
                    d_node['name'] = node.name

                # tag
                if node.tag is not None:
                    d_node['tag'] = node.tag

                # input_index
                try:
                    i = graph.input_nodes.index(node)
                    d_node['input_index'] = i
                except ValueError:
                    pass

                # output_index
                try:
                    i = graph.output_nodes.index(node)
                    d_node['output_index'] = i
                except ValueError:
                    pass

                #
                if node.obj_type is Graph:
                    subgraph, _ = graph.get_subgraph(node.tag)
                    d_node['input_variables'] = [id(_) for _ in subgraph.input_nodes]
                    d_node['output_variables'] = [id(_) for _ in subgraph.output_nodes]

                # data
                if isinstance(node, VariableGnode):
                    summary = node.data_collection.get_summary()
                    if len(summary) > 0:
                        d_node['data_summary'] = summary

                #
                nodes.append(d_node)
            for edge in graph.edges:
                d_edge = {
                    'source': id(edge.in_gnode),
                    'target': id(edge.out_gnode),
                    'arg_index': edge.arg_index,
                    'count': edge.count,
                }

                edges.append(d_edge)
        finally:
            graph.unlock()

        data = {
            'tag': graph.tag,
            'path': path,
            'nodes': nodes,
            'edges': edges,
            'input_variables': [id(_) for _ in graph.input_nodes],
            'output_variables': [id(_) for _ in graph.output_nodes],
        }
        json_data = json.dumps(data)
        return 'application/json', json_data

    if api_name == 'data' and method == 'GET':
        required = object()
        def read_query(key, type=str, default=required):
            if key in query:
                return type(query[key][0])
            elif default is required:
                raise ErrorResponse('Required query is missing: {}'.format(key))
            else:
                return default

        """
        data_index:
             None      ... all data
             'current' ... latest data
             int       ... index
        """

        data_name = read_query('name')
        data_index = read_query('index', default=None)
        data_type = read_query('type', default='json')

        # Get data
        try:
            node = get_obj(path)
        except KeyError:
            raise ErrorResponse('Invalid query path: {}'.format(path))

        if data_name not in node.data_collection:
            raise ErrorResponse('Invalid data name: {}'.format(data_name))
        data_series = node.data_collection[data_name]

        if data_index is None:
            # this `epochs` could include `None`
            epochs = data_series.get_iterations()
            epoch_list = []
            data_list = []
            for i, epoch in enumerate(epochs):
                data_index_ = i if epoch is not None else None
                epoch_, data_ = data_series.get_data(data_index_)
                epoch_list.append(epoch_)
                data_list.append(data_)
            data = {
                'epochs': epoch_list,
                'data': data_list,
            }
        else:
            data_index_ = int(data_index) if data_index != 'current' else None

            try:
                _, data = data_series.get_data(data_index_)
            except IndexError as e:
                raise ErrorResponse('Invalid data index: {}'.format(data_index))

        # Encode data
        if data is None:
            raise ErrorResponse('No data is available.')
        elif data_type == 'json':
            response_data = json.dumps(_ndarray_to_list(data))
            content_type = 'application/json'
        elif data_type == 'image':
            assert isinstance(data, numpy.ndarray) and data.ndim == 2
            import matplotlib
            import io
            dpi = 100
            width = 100
            height = 100
            figsize = (width / dpi, height / dpi)
            fig = matplotlib.pyplot.figure(figsize=figsize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.pcolormesh(data)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            matplotlib.pyplot.close(fig)
            response_data = buf.getvalue()
            content_type = 'image/png'
        else:
            raise ErrorResponse('Invalid data type: {}'.format(data_type))

        return content_type, response_data

    raise ErrorResponse('Invalid api request: {} method={} path={}'.format(
        api_name, method, path))


def pop_path(path):
    split = path.split('/', 1)
    if len(split) == 1:
        return split[0], ''
    else:
        return split[0], split[1]


def graph_app(environ, start_response):
    status = 200

    path = environ['PATH_INFO']
    path = path[1:] if path.startswith('/') else path
    root_path, path = pop_path(path)
    query = urllib.parse.parse_qs(environ['QUERY_STRING'])

    if root_path == 'api':
        api_name, path = pop_path(path)

        try:
            content_type, data = api(api_name, path, query, environ)
        except ErrorResponse as e:
            content_type = 'application/json'
            data = json.dumps({
                'error': str(e),
            })

        if isinstance(data, str):
            data = data.encode('utf-8')
    else:
        data = b'Error: Invalid request'
        content_type = 'text/plain'

    response = werkzeug.wrappers.Response(data)
    response.status_code = status
    response.headers['content-type'] = content_type
    response.headers['access-control-allow-origin'] = '*'
    return response(environ, start_response)
