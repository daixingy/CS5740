def read_csv(path):
    file_values = open(path,"r")
    reader= csv.reader(file_values)
    data= []
    for a in reader:
        if a[0] != 'id':
            data.append(a)
    return data

def compare(p1,p2):
	f1 = read_csv(p1)
	f2 = read_csv(p2)
	print(f1,f2)


if __name__ == '__main__':
	p1 = './data/propername/dev/dev_data.csv'
	p2 = './results/mlp_propername_test_predictions.csv'