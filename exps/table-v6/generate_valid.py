import codecs


def load_data(path):
    path_dict = {}

    with codecs.open(path, 'r', 'utf8') as fo:
        for line in fo:
            path_dict['/'.join(line.strip().split('/')[7:])] = None

    return path_dict

def dict_diff(paths1, path2):
    paths = []

    for path in paths1:
        if path in path2:
            paths.append(path)

    return paths

def write_data(paths, path, maindir):
    with open(path, 'w') as fw:
        for line in paths:
            fw.writelines((maindir + line + '\n').encode('utf8'))


paths1 = load_data('valid-v1.txt')
paths2 = load_data('valid-v2.txt')
valid_paths = dict_diff(paths1, paths2)
write_data(valid_paths, 'test-v1.txt', '/home/caory/github/darknet-master/datasets/table-v1/')
write_data(valid_paths, 'test-v2.txt', '/home/caory/github/darknet-master/datasets/table-v2/')
