import copy
list1 = [1, 2, 3, [4, 5]]
list2 = copy.deepcopy(list1)
list2.append(6)
list2[3].append(7)
print(id(list1))  # 输出：某个内存地址
print([id(ele) for ele in list1])
print(id(list2))  # 输出：与list1不同的内存地址
print([id(ele) for ele in list2])

