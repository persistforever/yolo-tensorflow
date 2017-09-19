import codecs


def load_data(path):
    path_dict = {}

    with codecs.open(path, 'r', 'utf8') as fo:
        for line in fo:
            path_dict['/'.join(line.strip().split('/')[7:])] = None

    return path_dict

def dict_diff(paths1, path2, path3):
    paths = []

    for path in paths1:
        if path in path2 and path in path3:
            paths.append(path)

    return paths

def write_data(paths, path, maindir):
    with open(path, 'w') as fw:
        for line in paths:
            fw.writelines((maindir + line + '\n').encode('utf8'))


paths1 = load_data('../../logs/table-v6/test-v1.txt')
paths2 = load_data('../../logs/table-v6/test-v2.txt')
paths4 = load_data('../../logs/table-v6/test-v4.txt')
valid_paths = dict_diff(paths1, paths2, paths4)
print(valid_paths)
write_data(valid_paths, '../../logs/table-v6/testing-v1.txt', '/home/caory/github/darknet-master/datasets/table-v1/')
write_data(valid_paths, '../../logs/table-v6/testing-v2.txt', '/home/caory/github/darknet-master/datasets/table-v2/')
write_data(valid_paths, '../../logs/table-v6/testing-v4.txt', '/home/caory/github/darknet-master/datasets/table-v4/')
