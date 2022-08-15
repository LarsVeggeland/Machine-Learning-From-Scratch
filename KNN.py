#---------- Imported libraries ----------

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

def read_data(filename : str) -> list:
    with open(filename, 'r') as file:
        file.readline()
        vectors = []
        for line in file.readlines():
            vector = [float(val) for val in line.split(',')]
            vectors.append(vector)
    
    return vectors


def euclidean(vector1 : list, vector2 : list) -> float:
    return sum((vector1[i]-vector2[i])**2 for i in range(len(vector1)))**0.5


def manhattan(vector1 : list, vector2 : list) -> float:
    return sum(abs(vector1[i]-vector2[i]) for i in range(len(vector1)))


def rescale(max_val, min_val, val) -> float:
    return (val -min_val)/(max_val - min_val)


def find_cols_max_min(data : list) -> list:
    res = []
    for i in range(len(data[0])-1):
        col = [row[i] for row in data]
        x_max = max(col)
        x_min = min(col)

        res.append([x_max, x_min])
    
    return res

def rescale_data(data : list, cols_max_min : list) -> list:

    rescaled = data.copy()

    for row in range(len(data)):
        for col in range(len(cols_max_min)):
            rescaled[row][col] = rescale(cols_max_min[col][0], cols_max_min[col][1], data[row][col])

    return rescaled


def KNN(classified_data : list, unclassified_data : list, K : int, distance, rescale : bool) -> list:

    if (K >= len(classified_data)):
        print(f"K too large at {K}. Cannot exceed classified data size at {len(classified_data)}")

    if (rescale):
        #cols_max_min = find_cols_max_min(cluster)
        #cluster = rescale_data(cluster, cols_max_min)
        #unclassified_data = rescale_data(unclassified_data, cols_max_min)
        scaler = MinMaxScaler()
        classified_data = scaler.fit_transform(classified_data)
        unclassified_data = scaler.fit_transform(unclassified_data)

    predictions = []

    for new_vector in unclassified_data:
        distances = []
        for old_vector in classified_data:
            distances.append([distance(new_vector[:-1], old_vector[:-1]), old_vector[-1]])

        distances = sorted(distances, key=lambda x : x[0])[:K]
        labels = [label[1] for label in distances]
        most_frequent = int(max(set(labels), key=labels.count))
        predictions.append(most_frequent)

    return predictions


def print_task_23():

    train = read_data("/Users/larsveggeland/Documents/Python/assignment2/train.csv")
    test = read_data("/Users/larsveggeland/Documents/Python/assignment2/test.csv")

    true_labels = [i[-1] for i in test]

    e = "euclidean"
    m = "manhattan"

    print("Not rescaled")

    print("K, Distance, Sample 1, Sample 2, Sample 3, Sample 4, Sample 5, Sample 6, Sample 7, Sample 8, Sample 9, Sample 10\n")
    for K in [1, 5, 9, 15]:
        for distance in [euclidean, manhattan]:
            res = KNN(train, test, K, distance, False)
            print(f"{K}, {e if  distance is euclidean else m}, {str(res)[1:-1]}")
            print(f"Accuracy {sum([res[i] == true_labels[i] for i in range(len(res))])/len(res) * 100} %")    
            cfm = confusion_matrix(true_labels, res).ravel().tolist()
            
            #print(f"{K}, {e if  distance is euclidean else m}, " + str(cfm)[1:-1] + "\n")
            print(f"Confusion matrix: (tn, fp, fn, tp)\n{cfm}\n")


def print_task_24():
    
    train = read_data("/Users/larsveggeland/Documents/Python/assignment2/train.csv")
    test = read_data("/Users/larsveggeland/Documents/Python/assignment2/test.csv")

    e = "euclidean"
    m = "manhattan"

    true_labels = [i[-1] for i in test]

    print("\nRescaled")
    with open("KNN_rescaled_CM.csv", "w") as file:
        file.write("K, Distance, TN, FP, FN, TP\n")
    with open("KNN_rescaled.csv", "w") as csv:
        csv.write("K, Distance, Sample 1, Sample 2, Sample 3, Sample 4, Sample 5, Sample 6, Sample 7, Sample 8, Sample 9, Sample 10\n")
        for K in [1, 5, 9, 15]:
            for distance in [euclidean, manhattan]:
                res = KNN(train, test, K, distance, True)
                csv.write(F"{K}, {e if  distance is euclidean else m}, {str(res)[1:-1]}\n")
                #print(f"K: {K}, distance: {e if  distance is euclidean else m}, prediction:\n{res}\n")
                with open("KNN_rescaled_CM.csv", "a") as file:
                    file.write(str())
                cfm = confusion_matrix(true_labels, res).ravel().tolist()
                with open("KNN_rescaled_CM.csv", "a") as file:
                    file.write(f"{K}, {e if  distance is euclidean else m}, " + str(cfm)[1:-1] + "\n")
                print(f"\nConfusion matrix: (tn, fp, fn, tp)\n{cfm}\n")


print_task_23()

