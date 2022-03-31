import pickle
import matplotlib.pyplot as plt
import os

def data_make(path, name = ""):
    data = {}
    labels = []
    with open(path, 'r') as f:
        line = f.readline()
        for word in line.split():
            labels.append(word) 
    
    with open(path, 'r') as f:
        lines = f.readlines()[1:]
        word_counter = 0
        for line in lines:
            for word in line.split():
                if word == "\n": continue
                key = labels[word_counter]
                if key not in data:
                    data[key] = []
                data[key].append(float(word))
                word_counter = word_counter + 1
                if word_counter == len(labels): word_counter = 0

    if name == "": name = path[path.find("/"):]
    with open("dicts/"+name, 'wb') as f:
        pickle.dump(dict, f)
    
    return data

def plot_make(data, dir_name, x = 'n'):
    path = os.getcwd() + f"/plots/{dir_name}"
    try: os.mkdir(path)
    except OSError: print("fail")

    for y in data.keys():
        if y != x: 
            plt.plot((data[x]), (data[y]))
            plt.suptitle(f"{y}")
            plt.savefig(f"{path}/{y}")          
            plt.clf()  
    for y in data.keys():
        print(y)
        if "time" in y:
            plt.plot((data[x]), (data[y]), label= f"{y[:y.find('_')]}")
            plt.legend(loc="upper left")
            plt.suptitle(f"time")
            plt.savefig(f"{path}/time")
    plt.clf()     
    for y in data.keys():      
        if "lenght" in y:
            plt.plot((data[x]), (data[y]), label= f"{y[:y.find('_')]}")
            plt.legend(loc="upper left")
            plt.suptitle(f"lenght")
            plt.savefig(f"{path}/lenght")                  
    plt.clf()
    return
def plot_all():
    directory = 'results'
    for filename in os.listdir(directory):
        data = data_make(directory+'/'+filename)
        plot_make(data,filename)


plot_all()
#data = data_make("results/test_random_time_Symmetric_2-100.txt")
#plot_make(data,"test_random_time_Symmetric_2-100.txt")