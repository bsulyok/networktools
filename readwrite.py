from common import edge_iterator
import csv

def n_to_data(n):
    if n <= 62:
        return [n]
    elif n <= 258047:
        return (63, (n >> 12) & 63, (n >> 6) & 63, n & 63)
    else:
        return (63, 63, (n >> 30) & 63, (n >> 24) & 63, (n >> 18) & 63, (n >> 12) & 63, (n >> 6) & 63, n & 63)

def data_to_n(data):
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    return (data[2] << 30) + (data[3] << 24) + (data[4] << 18) + (data[5] << 12) + (data[6] << 6) + data[7], data[8:]

def graph_parser(adjacency_list):
    yield b'>>sparse6<<'
    yield b':'
    N = len(adjacency_list)
    for d in n_to_data(N):
        yield str.encode(chr(d + 63))
    k = 1
    while 1 << k < N:
        k += 1
    def enc(x):
        return [1 if (x & 1 << (k-1-i)) else 0 for i in range(k)]
    bits = []
    #current_vertex = next(iter(adjacency_list))
    current_vertex = None
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if neighbour < vertex:
            if current_vertex == vertex:
                bits.append(0)
                bits.extend(enc(neighbour))
            else:
                current_vertex = vertex
                bits.append(1)
                bits.extend(enc(vertex))
                bits.append(0)
                bits.extend(enc(neighbour))
    if k < 6 and N == (1 << k) and ((-len(bits)) % 6) >= k and current_vertex < (N - 1):
        bits.append(0)
    bits.extend([1] * ((-len(bits)) % 6))
    for i in range(0, len(bits), 6):
        yield str.encode(chr((bits[i + 0] << 5) + (bits[i + 1] << 4) + (bits[i + 2] << 3) + (bits[i + 3] << 2) + (bits[i + 4] << 1) + (bits[i + 5] << 0) + 63))
    yield b'\n'

def data_parser(string):
    chars = [c-63 for c in string[12:]]
    N, data = data_to_n(chars)
    k = 1
    while 1 << k < N:
        k += 1
    def parse_data():
        chunks = iter(data)
        d = None
        dlen = 0
        while True:
            if dlen < 1:
                try:
                    d = next(chunks)
                except:
                    return
                dlen = 6
            dlen -= 1
            b = (d >> dlen) & 1
            x = d & ((1 << dlen ) - 1)
            xlen = dlen
            while xlen < k:
                try:
                    d = next(chunks)
                except:
                    return
                dlen = 6
                x = ( x << 6) + d
                xlen += 6
            x = x >> (xlen - k)
            dlen = xlen - k
            yield b, x

    #adjacency_list = {vertex:{} for vertex in range(N)}
    adjacency_list = {}
    v = None
    for b, u in parse_data():
        if b == 1:
            v = u
        elif b == 0:
            try:
                adjacency_list[u].update({v:{}})
            except:
                adjacency_list[u] = {}
                adjacency_list[u].update({v:{}})
            try:
                adjacency_list[v].update({u:{}})
            except:
                adjacency_list[v] = {}
                adjacency_list[v].update({u:{}})
    return adjacency_list

def write_adjacency_list(adjacency_list, filename='test.txt'):
    with open(filename, mode='wb') as outfile:
        for b in graph_parser(adjacency_list):
            outfile.write(b)

def read_adjacency_list(filename='test.txt'):
    with open(filename, mode='rb') as infile:
        string = infile.read()
    return data_parser(string)

def csv_feeder(adjacency_list, attributes):
    yield ["vertex=<class 'int'>", "neighbour=<class 'int'>"] + [attr + '=' + str(attr_type) for attr, attr_type in attributes.items()]
    for vertex, neighbour, attr in edge_iterator(adjacency_list):
        if neighbour < vertex:
            yield [vertex, neighbour] + [attr.get(attribute) for attribute in attributes]

def write_graph(adjacency_list, filename, attributes):
    with open(filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(csv_feeder(adjacency_list, attributes))

def infer_type(attr_type):
    if attr_type == "<class 'int'>":
        return int
    elif attr_type == "<class 'float'>":
        return float
    elif attr_type == "<class 'str'>":
        return str
    elif attr_type == "<class 'complex'>":
        return complex
    
def read_graph(filename):
    with open(filename, mode='r', newline='') as infile:
        reader = csv.reader(infile, delimiter=',')
        attributes = {}
        for label in next(reader).split(',')[2:]:
            attr, attr_type = label.split('=')
            attributes[attr] = infer_type(attr_type)
        adjacency_list = {}
        for row in reader:
            (vertex, neighbour), attr = row[:2], row[2:]
            vertex, neighbour, attr = int(row[0]), int(row[1]), row[2:]
            if neighbour not in adjacency_list:
                adjacency_list[neighbour] = {}
            if vertex not in adjacency_list:
                adjacency_list[vertex] = {}
            adjacency_list[vertex][neighbour] = {attribute:attribute_type(attr[idx]) for idx, (attribute, attribute_type) in enumerate(attr.items())}
            adjacency_list[vertex][neighbour] = {attributes[i]:attribute for i, attribute in enumerate(attr)}
            adjacency_list[neighbour][vertex] = {attributes[i]:attribute for i, attribute in enumerate(attr)}
            print(row)
    return adjacency_list