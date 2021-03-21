import csv
def read_csv(path):
    file_values = open(path,"r")
    reader= csv.reader(file_values)
    data= []
    for a in reader:
        if a[0] != 'id':
            data.append(a[1])
    return data

def compare(p1,p2):
    f1 = read_csv(p1)
    f2 = read_csv(p2)
    dict = {}
    for i,j in zip(f1,f2):
        if i not in dict:
            dict[i] = [["person",0],["place",0],["movie",0],["drug",0],["company",0]]

        for k in range(len(dict[i])):
            if dict[i][k][0] == j:
                dict[i][k][1] += 1
    for i in dict:
        print(i,dict[i])
       
    s = 0
    for i in dict:
        for j in dict[i]:
            s += j[1]
    print(s)




if __name__ == '__main__':
    p1 = './data/propername/dev/dev_labels.csv'
    p2 = './results/mlp_propername_dev_predictions.csv'
    compare(p1,p2)