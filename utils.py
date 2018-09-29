def print_dic(dic):
    for key in dic.keys():
            if isinstance(dic[key], dict):
                print(key, ":")
                for items in dic[key]:
                    print("    %s : %s" % (items, dic[key][items]))
            else:
                print(key, ':', dic[key])