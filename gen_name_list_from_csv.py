FILE_NAME = 'Sheet2.csv'


def main():
    with open(FILE_NAME) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]

    name_set = set()

    for line in lines:
        names_of_line = line.split(',')
        names_of_line = list(filter(None, names_of_line))
        names_of_line = [x.strip() for x in names_of_line]
        name_set.update(names_of_line)
    name_set = sorted(name_set)

    with open('name_list', 'a') as f:
        for name in name_set:
            f.write('{}\n'.format(name))


if __name__ == '__main__':
    main()
