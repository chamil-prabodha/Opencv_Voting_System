
arr = [3,4,1,5,2,6]
temp = [0]*len(arr)

print(arr)


def mergeSort(a,p,r):

    if p<r:
        q = (p+r)/2
        mergeSort(a,p,q)
        mergeSort(a,q+1,r)
        merge(a,p,q,r)

def merge(a,p,q,r):

    i=p
    while i<=r:
        temp[i] = a[i]
        i+=1

    i = p
    j= q+1
    k = p

    while i<=q and j<=r:
        if temp[i] < temp[j]:
            a[k] = temp[i]
            i+=1
        else:
            a[k] = temp[j]
            j+=1

        k+=1

    while (i<=q):
        a[k] = temp[i]
        i+=1
        k+=1

def bubbleSort(a):

    i = len(a)-1
    while i>=0:
        for j in range(1,len(a)):
            if a[j]<a[j-1]:
                temp = a[j]
                a[j] = a[j-1]
                a[j-1] = temp
        i-=1
def selectionSort(a):

    for i in range(0,len(a)-1):
        min = i
        for j in range(i+1,i+1):
            if a[j] < a[min]:
                min = j
        temp = a[i]
        a[i] = a[min]
        a[min] = temp

def insertionSort(a):

    for i in range(1,len(a)):
        index = a[i]
        j=i;
        while j>0 and a[j-1]>index:
            a[j] = a[j-1]
            j-=1
        a[j] = index


# mergeSort(arr,0,len(arr)-1)
bubbleSort(arr)
print(arr)