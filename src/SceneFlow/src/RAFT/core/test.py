import sys

class Optical:
    
    def __init__(self):
        print ('Initializing RAFT network...')
        
    def GetFlow(self, a):
        a +=1
        return a

def optical(a):
    print("***")
    # if(image1.size == 0):
    #     print("0")
    # a = image1.shape
    print("***")
    # print(a)
    
    op = Optical()
    result = op.GetFlow(a)
    print(result)
    return result
    
    

if __name__ == "__main__":
    
    optical()