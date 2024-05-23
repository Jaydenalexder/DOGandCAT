import shutil
import os

'''
主要是用来将一个文件夹中的猫和狗的图片
分别移动到训练集和验证集的不同文件夹中。具体实现是通过遍历文件夹中的所有文件，
根据文件名中的"cat"或"dog"关键字来判断是猫还是狗的图片，
然后将其移动到对应的文件夹中。同时，还会删除一些多余的图片。
'''
 
def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    # print(filelist)
    cat_n =0
    dog_n = 0
    for file in filelist:
        src = os.path.join(old_path, file)
        if not os.path.isfile(src):
            continue
        animal_str = str(file).split('.')[0]
        if animal_str =='cat': 
            if cat_n < 2500:
                cat_path = os.path.join('./validation/cat/', file)
                shutil.move(src, cat_path)
                cat_n +=1
            elif cat_n < 12500:
                cat_path = os.path.join('./train/cat/', file)
                shutil.move(src, cat_path)
                cat_n += 1
            else:
                os.remove(src)
        elif animal_str == 'dog':
            if dog_n < 2500:
                dog_path = os.path.join('./validation/dog/', file)
                shutil.move(src, dog_path)
                dog_n += 1
            elif dog_n < 12500:
                dog_path = os.path.join('./train/dog/', file)
                dog_n += 1
                shutil.move(src, dog_path)
            else:
                os.remove(src)
        else:
            continue
        
'''
用于将猫和狗的图片分类到不同的文件夹中。
遍历一个文件列表，如果文件不是一个文件，则跳过。
然后，它将文件名中的动物字符串提取出来，
如果是猫，则将文件移动到验证或训练文件夹中的相应子文件夹中，
如果是狗，则执行相同的操作
。如果动物字符串不是猫或狗，则跳过。
如果猫或狗的数量超过了一定的限制，则删除文件。        
'''
if __name__ == '__main__':
    os.makedirs('./train/cat/')
    os.makedirs('./train/dog/')
    os.makedirs('./validation/cat/')
    os.makedirs('./validation/dog/')
    remove_file(r"./train/", r"./validation/")
