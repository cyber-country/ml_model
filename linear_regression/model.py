def train_model():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error,r2_score
    from sklearn.preprocessing import StandardScaler
    #Data extraction + data cleaning with empty value
    x=[]
    y=[]
    with open("houses.csv") as m:
        m.readline()
        while True:
            data=m.readline()
            if data=="":
                break
            data=data.strip()
            temp=data.split(",")
            if "NA" in temp or "" in temp:
                continue
            x.append([float(temp[1]),float(temp[0]),float(temp[2]),float(temp[3]),float(temp[4])])
            y.append(float(temp[17]))
    x=np.array(x)
    y=np.array(y)
    #data split to avoid overfitting
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=120)
    #scaler
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(xtrain)
    xtest=scaler.transform(xtest)
    #model trainning
    model=LinearRegression()
    model.fit(xtrain,ytrain)
    #model performance
    ypred=model.predict(xtest)
    mse=mean_squared_error(ytest,ypred)
    r2=r2_score(ytest,ypred)
    return model,scaler,mse,r2