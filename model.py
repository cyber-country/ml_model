def model_train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,confusion_matrix
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    x=[]
    y=[]
    with open("student_exam_data.csv") as f:
        f.readline()
        while True:
            data=f.readline()
            if data=="":
                break
            data=data.strip()
            temp=data.split(",")
            if "NA" in temp or "" in temp:
                continue
            x.append([float(temp[0]),float(temp[1])])
            y.append(float(temp[2]))
    #Train-split-test
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1000)
    #scaler
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(xtrain)
    xtest=scaler.transform(xtest)
    #model training
    model=LogisticRegression()
    model.fit(xtrain,ytrain)
    r=model.predict_proba(xtest)
    y_pred=[]
    for p in r:
        if p[1]>=0.49999:
            y_pred.append(1)
        else:
            y_pred.append(0)
    accuracy=accuracy_score(y_pred,ytest)
    c=confusion_matrix(ytest,y_pred)
    return model,scaler,accuracy,c