#!/usr/bin/env python
# coding: utf-8

# In[173]:


print(confusion_matrix)


# In[ ]:





# 5(A) The perceptron classifier has the highest accuracy for examples that belong to class 5 with an accuracy of roughly 80%

# 5(B) The perceptron classifier has the lowest accuracy for examples that belong to class 3 with an accuracy of roughly 37%

# 5(C):  Perceptron 1 predicts 3 the worst, Perceptron 2 predicts 3 the worst, 3 predicts 1 the worst, 6 predict 5 the worst, 4 predicts 1 the worst, and 5 predicts 6 the worst

# In[128]:


import numpy as np


# In[129]:


train_data = np.loadtxt('pa3train.txt')
test_data = np.loadtxt('pa3test.txt')
dictionary = np.loadtxt('pa3dictionary.txt',dtype = str)


# In[130]:


X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:,:-1]
y_test = test_data[:,-1]


# In[131]:


print(X_train)
train_data


# In[132]:


X_train_1_or_2 = train_data[np.logical_or(y_train ==1, y_train ==2), :-1]
y_train_1_or_2 = train_data[np.logical_or(y_train ==1, y_train ==2), -1]
X_test_1_or_2 = test_data[np.logical_or(y_test==1, y_test==2), :-1]
y_test_1_or_2 = test_data[np.logical_or(y_test==1, y_test==2), -1]
#how to get the subset of indices where label is either 1 or 2


#as the perceptron deals in positives and negatives must convert
#one of the labels to a negative, I have picked y_train = 2 to be converted
size_current_y_train = len(y_train_1_or_2)
size_current_y_test = len(y_test_1_or_2)
for i in range(size_current_y_train):
    if y_train_1_or_2[i] == 2:
        y_train_1_or_2[i] = -1.0
        #make the selected label a negative, perfect to have one pos
        #and one neg for perceptron's binary classifier algorithm
        
for i in range(size_current_y_test):
    if y_test_1_or_2[i] == 2:
        y_test_1_or_2[i] = -1.0


# In[133]:


def calculate_linear_error(weights, x,y):
    num_mistakes = 0
    
    predicted = None
    x_size = len(x)
    
    for i in range(x_size):
        if np.dot(weights,x[i]) <0:
            
            predicted = -1
        else:
            
            predicted = 1
            
        if predicted != y[i]:
            #if the predicted label of positive for =1 or negative for
            #equals 2 and the actual label is the opposite, then the linear classification was incorrect 
            num_mistakes +=1
    percent_mistakes = num_mistakes / x_size
    return percent_mistakes


# Part 1

# In[134]:


train_vec_len = len(X_train_1_or_2[1])
weights  = np.zeros(train_vec_len)
def perceptron_alg(weights,vec,labels):
    vec_size = len(vec)
    for i in range(vec_size):
        if labels[i]*np.dot(weights,vec[i]) <= 0:
            weights = weights +labels[i] *vec[i]
    return weights


# In[135]:


w1_output = perceptron_alg(weights, X_train_1_or_2, y_train_1_or_2)
print('The training error after the first pass is: ' + str(calculate_linear_error(w1_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the first pass is: ' + str(calculate_linear_error(w1_output, X_test_1_or_2, y_test_1_or_2)))
print()

w2_output = perceptron_alg(w1_out, X_train_1_or_2, y_train_1_or_2)
print('The training error after the second pass is: ' + str(calculate_linear_error(w2_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the second pass is: ' + str(calculate_linear_error(w2_output, X_test_1_or_2, y_test_1_or_2)))
print()

w3_output = perceptron_alg(w2_out, X_train_1_or_2, y_train_1_or_2)
print('The training error after the third pass is: ' + str(calculate_linear_error(w3_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the third pass is: ' + str(calculate_linear_error(w3_output, X_test_1_or_2, y_test_1_or_2)))
print()

w4_output = perceptron_alg(w3_out, X_train_1_or_2, y_train_1_or_2)
print('The training error after the fourth pass is: '+ str(calculate_linear_error(w4_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the fourth pass is: ' + str(calculate_linear_error(w4_output, X_test_1_or_2, y_test_1_or_2)))
print()


# Part 2

# In[136]:


train_vec_len = len(X_train_1_or_2[1])
weights = np.zeros(train_vec_len)
def logisticregress(i, weights, vec, labels, lr = .001):
    for iter_count in range(i):

        loss_function = 0
        for i in range(len(vec)):
        
            exponent = labels[i] * np.dot(weights.T, vec[i])
            denominator = 1+ np.exp(exponent)
        
            numerator = vec[i] * labels[i]
        
            loss_function += numerator / denominator
        weights = weights + lr*loss_function
    return weights


# In[137]:


iter_10_output = logisticregress(10, weights, X_train_1_or_2, y_train_1_or_2)

iter_50_output = logisticregress(50, weights, X_train_1_or_2, y_train_1_or_2)

iter_100_output = logisticregress(100, weights, X_train_1_or_2, y_train_1_or_2)

iter_2_output = logisticregress(2, weights, X_train_1_or_2, y_train_1_or_2)


# In[138]:


print('The Training Error after 10 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_10_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the 10 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_10_output, X_test_1_or_2, y_test_1_or_2)))
print()
print()
print('The Training Error after 50 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_50_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the 50 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_50_output, X_test_1_or_2, y_test_1_or_2)))
print()
print()
print('The Training Error after 100 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_100_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the 100 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_100_output,X_test_1_or_2, y_test_1_or_2)))

print()
print()
print('The Training Error after 2 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_2_output, X_train_1_or_2, y_train_1_or_2)))
print('The test Error after the 2 Iterations of Gradient Descent is: ' + str(calculate_linear_error(iter_2_output, X_test_1_or_2, y_test_1_or_2)))
#According to the PA instructions the code is right if after 2 iterations
#the train error is approximately .497, which is basically what I got


# Part 3

# In[139]:


sort_w4 = sorted(w4_output)
size_w4_output = len(w4_output)
max_values = set(sort_w4[-3:])
max_val_coord = []

min_values = set(sort_w4[:3])
min_val_coord = []

for i in range(size_w4_output):
    if w4_output[i] in max_values:
        max_val_coord.append(i)
    if w4_output[i] in min_values:
        min_val_coord.append(i)


# In[140]:


print('The words that most strongly indicate the positive class are the ')
print(str(max_val_coord[0]) + 'th coordinate, with a value = ' + str(w4_output[max_val_coord[0]]) + ', and the Dictionary Word being:  ' + dictionary[max_val_coord[0]])
print(str(max_val_coord[1]) + 'th coordinate, with a value = ' + str(w4_output[max_val_coord[1]]) + ', and the Dictionary Word being: ' + dictionary[max_val_coord[1]])
print(str(max_val_coord[2]) + 'th coordinate, with a value = ' + str(w4_output[max_val_coord[2]]) + ', and the Dictionary Word being:  ' + dictionary[max_val_coord[2]])
print()
print()

print('The words that most strongly indicate the negative class are the')
print(str(min_val_coord [0]) + 'th coordinate, with a value =' + str(w4_output[min_val_coord[0]]) + ', and the Dictionary Word being:   ' + dictionary[min_val_coord[0]])
print(str(min_val_coord [1]) + 'th coordinate, with a value = ' + str(w4_output[min_val_coord[1]]) + ', and the Dictionary Word being: ' + dictionary[min_val_coord[1]])
print(str(min_val_coord [2]) + 'th coordinate, with a value = ' + str(w4_output[min_val_coord[2]]) + ', and the Dictionary Word being: ' + dictionary[min_val_coord[2]])


# Part 4

# In[141]:


sort_iter_50 = sorted(iter_50_output)
size_iter_50_output = len(iter_50_output)
max_values = set(sort_iter_50[-3:])
max_val_coord = []

min_values = set(sort_iter_50[:3])
min_val_coord = []

for i in range(size_iter_50_output):
    if iter_50_output[i] in max_values:
        max_val_coord.append(i)
    if iter_50_output[i] in min_values:
        min_val_coord.append(i)


# In[142]:


print('The words that most strongly indicate the positive class are the ')
print(str(max_val_coord[0]) + 'th coordinate, with a value = ' + str(iter_50_output[max_val_coord[0]]) + ', and the Dictionary Word being:  ' + dictionary[max_val_coord[0]])
print(str(max_val_coord[1]) + 'th coordinate, with a value = ' + str(iter_50_output[max_val_coord[1]]) + ', and the Dictionary Word being: ' + dictionary[max_val_coord[1]])
print(str(max_val_coord[2]) + 'th coordinate, with a value = ' + str(iter_50_output[max_val_coord[2]]) + ', and the Dictionary Word being:  ' + dictionary[max_val_coord[2]])
print()
print()

print('The words that most strongly indicate the negative class are the')
print(str(min_val_coord [0]) + 'th coordinate, with a value =' + str(iter_50_output[min_val_coord[0]]) + ', and the Dictionary Word being:   ' + dictionary[min_val_coord[0]])
print(str(min_val_coord [1]) + 'th coordinate, with a value = ' + str(iter_50_output[min_val_coord[1]]) + ', and the Dictionary Word being: ' + dictionary[min_val_coord[1]])
print(str(min_val_coord [2]) + 'th coordinate, with a value = ' + str(iter_50_output[min_val_coord[2]]) + ', and the Dictionary Word being: ' + dictionary[min_val_coord[2]])


# Part 5

# In[143]:


train_vec_len = len(X_train_1_or_2[1])
weights = np.zeros(train_vec_len)

y_1 = np.copy(y_train)
y_2 = np.copy(y_train)
y_3 = np.copy(y_train)
y_4 = np.copy(y_train)
y_5 = np.copy(y_train)
y_6 = np.copy(y_train)
y_1[y_1 != 1] = -1

y_2[y_2 != 2] = -1
y_2[y_2 == 2] = 1

y_3[y_3 != 3] = -1
y_3[y_3 == 3] = 1

y_4[y_4 != 4] = -1
y_4[y_4 == 4] = 1

y_5[y_5 != 5] = -1
y_5[y_5 == 5] = 1

y_6[y_6 != 6] = -1
y_6[y_6 == 6] = 1

confusion_matrix = np.zeros((7,6))


# In[ ]:





# In[144]:


one_pass_on_all_weights = [perceptron_alg(weights, X_train, y_1), perceptron_alg(weights, X_train, y_2), perceptron_alg(weights, X_train, y_3),
        perceptron_alg(weights, X_train, y_4), perceptron_alg(weights, X_train, y_5), perceptron_alg(weights, X_train, y_6)]
one_pass_on_all_weights


# In[145]:


size_X_test = len(X_test)

size_all_weights = len(one_pass_on_all_weights)
for i in range(size_X_test):
    total_predicts = np.zeros(size_all_weights)
    
    for w in range(size_all_weights):
        
        
        if np.dot(one_pass_on_all_weights[w], X_test[i]) > 0:
            total_predicts[w] = 1
            #binary decider like perceptron
        else:
            total_predicts[w] = -1
    if (np.mean(total_predicts == 1)) != (1/6):
        #in this case, the dont know clause is used, as the prediction is succesful
        #only as long as a single value is positive and all the rest are negative
        #in the event that there's more than 1 positive we can't be certain that a
        #prediction should be made
        predicted_index =6 #where to put values with an i dont know result, the 7th row
    else:
        predicted_index = np.where(total_predicts ==1)[0][0]
        #in this case, index is stored when only a single positive value is
        #reached 
    confusion_matrix[predicted_index, int(y_test[i] -1)] += (1/np.sum(y_test == y_test[i]))
#have to put int with (y_test[i]-1) or else t's invalid indices


# In[146]:


print(confusion_matrix)


# In[ ]:





# In[ ]:





# In[ ]:




