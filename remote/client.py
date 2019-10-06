import socket
from contextlib import contextmanager
from .sexpdata import loads, dumps


class RemoteClient:
    '''A thin wrapper around socket client APIs'''

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def disconnect(self):
        self.__raw_send('Stop')
        self.sock.close()

    def __raw_send(self, msg):
        # The server side won't terminate parsing until it sees a '\n'
        self.sock.sendall(str.encode(msg + '\n'))

    def __raw_recv(self):
        msg = ''
        while True:
            # Server side will use '\n' as message terminator
            raw_chunk = self.sock.recv(4096)
            if not raw_chunk:
                raise IOError('Remote server disconnected')
            chunk = raw_chunk.decode('utf-8')
            msg += chunk
            if msg.endswith('\n'):
                break
        return msg.strip()

    def __communicate(self, msg):
        self.__raw_send(msg)
        reply_msg = self.__raw_recv()
        try:
            reply_sexp = loads(reply_msg)
            if len(reply_sexp) == 0:
                raise IOError(
                    'Unexpected parsing result of messsage "{}"'.format(reply_msg))
            reply_sexp[0] = reply_sexp[0].value()
            if reply_sexp[0] == 'Error':
                raise IOError(reply_sexp[1])
            return reply_sexp
        except AssertionError:
            raise IOError(
                'Sexp parsing error for message "{}"'.format(reply_msg))
        except IndexError:
            raise IOError(
                'Sexp index out of bound for message "{}"'.format(reply_msg))

    def __expect_ack(self, resp):
        try:
            if resp[0] != 'Ack':
                raise IOError('Protocol error: {}'.format(dumps(resp)))
            return int(resp[1])
        except IndexError:
            raise IOError('Protocol error: {}'.format(dumps(resp)))

    def __expect_ackstring(self, resp):
        try:
            if resp[0] != 'AckString':
                raise IOError('Protocol error: {}'.format(dumps(resp)))
            return resp[1].value()
        except IndexError:
            raise IOError('Protocol error: {}'.format(dumps(resp)))

    def __expect_state(self, resp):
        try:
            ret = dict()
            if resp[0] == 'NextState':
                ret['features'] = resp[1]
                ret['available_actions'] = resp[2]
                ret['is_final'] = False
            elif resp[0] == 'Reward':
                ret['reward'] = float(resp[1])
                ret['is_final'] = True
            else:
                raise IOError('Protocol error: {}'.format(dumps(resp)))
            return ret
        except IndexError:
            raise IOError('Protocol error: {}'.format(dumps(resp)))

    def get_num_actions(self):
        resp = self.__communicate('ActionCount')
        return self.__expect_ack(resp)

    def get_num_features(self):
        resp = self.__communicate('FeatureCount')
        return self.__expect_ack(resp)

    def get_num_training(self):
        resp = self.__communicate('TrainingBenchCount')
        return self.__expect_ack(resp)

    def get_num_testing(self):
        resp = self.__communicate('TestingBenchCount')
        return self.__expect_ack(resp)

    def get_action_name(self, idx):
        resp = self.__communicate('(ActionName {})'.format(idx))
        return self.__expect_ackstring(resp)

    def get_bench_name(self, idx):
        resp = self.__communicate('(BenchName {})'.format(idx))
        return self.__expect_ackstring(resp)

    def start_rollout(self, idx):
        resp = self.__communicate('(PickBench {})'.format(idx))
        return self.__expect_state(resp)

    def restart_rollout(self):
        resp = self.__communicate('RestartBench')
        return self.__expect_state(resp)

    def take_action(self, idx):
        resp = self.__communicate('(TakeAction {})'.format(idx))
        return self.__expect_state(resp)


@contextmanager
def open_connection(*args, **kwargs):
    sock = socket.create_connection(*args, **kwargs)
    client = RemoteClient(sock)
    yield client
    client.disconnect()
