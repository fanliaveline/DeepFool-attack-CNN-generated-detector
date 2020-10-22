import os
class BatchRename():
 
    def __init__(self):
        self.path = r'/home/yanxu-weili/Leslie/CNNDetection-master/dataset/test/cyclegan/winter/0_real'  #表示需要命名处理的文件夹

    def rename(self):
        os.listdir() #方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(self.path)
        total_num = len(filelist) #获取文件夹内所有文件个数
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.png'):
            #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),str(i) +'b'+ '.png')
                try:
                    os.rename(src, dst)
                   #print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d pngs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
