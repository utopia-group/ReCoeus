import logging
import socket
from .sexpdata import loads, dumps


LOG = logging.getLogger(__name__)


def raw_recv(sock):
    msg = ''
    while True:
        # Server side will use '\n' as message terminator
        raw_chunk = sock.recv(4096)
        if not raw_chunk:
            raise IOError('Remote client is disconnected')
        chunk = raw_chunk.decode('utf-8')
        msg += chunk
        if msg.endswith('\n'):
            break
    return msg.strip()


def raw_send(sock, msg):
    # The server side won't terminate parsing until it sees a '\n'
    sock.sendall(str.encode(msg + '\n'))


def get_response(handler, request_sexp):
    if request_sexp[0] == 'Error':
        raise IOError(request_sexp[1])
    elif request_sexp[0] == 'NextState':
        state_dict = dict()
        state_dict['features'] = request_sexp[1]
        state_dict['available_actions'] = request_sexp[2]

        num_actions = len(request_sexp[2])
        prio_distr = handler(state_dict)
        if not isinstance(prio_distr, list):
            raise IOError('Protocol error: '
                          'server handler must return a list of numbers: {}'
                          .format(prio_distr))
        if len(prio_distr) != num_actions:
            raise IOError(
                'Protocol error: distribution contains {} items, but there are {} actions'.format(
                    num_actions, len(prio_distr)))
        resp_msg = '(Probability ({}))'.format(
            ' '.join([str(x) for x in prio_distr]))
        return resp_msg
    else:
        raise IOError('Protocol error: {}'.format(dumps(resp_msg)))


def communicate(sock, handler):
    request_msg = raw_recv(sock)
    request_sexp = loads(request_msg)
    if len(request_sexp) == 0:
        raise IOError(
            'Unexpected parsing result of messsage "{}"'.format(request_msg))
    request_sexp[0] = request_sexp[0].value()
    if request_sexp[0] == 'Error':
        raise IOError(request_sexp[1])
    elif request_sexp[0] == 'Stop':
        return True
    response = get_response(handler, request_sexp)
    if response is not None:
        raw_send(sock, response)
    return False


def establish_simple_server(addr, port, handler):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (addr, port)
    sock.bind(server_address)
    sock.listen(0)

    try:
        while True:
            # Wait for a connection
            connection, client_address = sock.accept()
            try:
                LOG.info('connection from {}'.format(client_address))
                # Receive the data in small chunks and retransmit it
                while True:
                    should_stop = communicate(connection, handler)
                    if should_stop:
                        break
            except AssertionError as e:
                LOG.warning('Sexp parsing error: {}'.format(e))
            except IndexError as e:
                LOG.warning('Sexp index out of bound error: {}'.format(e))
            except IOError as e:
                LOG.warning('I/O error: {}'.format(e))
            finally:
                # Clean up the connection
                LOG.info('connection closed')
                connection.close()
    except KeyboardInterrupt:
        LOG.warning('Received stop request from user.')
    finally:
        sock.close()
