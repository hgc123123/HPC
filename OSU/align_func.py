import sys
def addFile(filename1,filename2,filename3):
    # print('hello')
    filename1=open(filename1,'r')
    filename2=open(filename2,'r')
    filename3=open(filename3,'w')
    i = 0
    while i < 4:
        line1 = filename1.readline()
        line2 = filename2.readline()
        i += 1
    while line1:
        #filename3.write(line1.split('\n')+"\t"+line2.strip('\n').split()[1])
        filename3.write(line1.split('\n')[0])
        filename3.write('\t\t')
        filename3.write("{0:>10}".format(line2.strip('\n').split()[1]))
        filename3.write('\n')
        #print(line1.strip('\n').split()[1])
        line1=filename1.readline()
        line2=filename2.readline()


if __name__ == '__main__':
    filename1=sys.argv[1]
    filename2=sys.argv[2]
    filename3=sys.argv[3]
    addFile(filename1,filename2,filename3)
