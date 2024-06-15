def dumpclean(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k)
                dumpclean(v)
            else:
                print('%s : %s' % (k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print(v)
    else:
        print(obj)


confusion = {
    "TrueNegative": conf_mat[0][0],
    "FalseNegative": conf_mat[0][1],
    "FalsePositive": conf_mat[1][0],
    "TruePositive": conf_mat[1][1],
}
confusion["TrueNegative"]
confusion



dumpclean(confusion)