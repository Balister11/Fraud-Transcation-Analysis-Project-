from flask import Flask , render_template , request
import numpy as np 
import pickle
import sklearn


from sklearn.linear_model import LogisticRegression

model=pickle.load(open('E:\\19-02-2023,Handling outliers, EDA , STD, IQR , Categorical data\\Classification Project\\credit_project.pkl','rb'))
standard_scaler=pickle.load(open('E:\\19-02-2023,Handling outliers, EDA , STD, IQR , Categorical data\\Classification Project\\standardscaler.pkl','rb'))
app=Flask(__name__)

@app.route('/')

def fun():
    return render_template('index.html')

@app.route('/predict',methods=['Get','Post'])
def predict():

    
    a=[i for i in request.form.values()]

    a = [int(j) if j.isdigit() else float(j) for j in a]

    a = np.array([a])
    res=standard_scaler.transform(a)
    sol=model.predict(res)[0]
    if sol==0:
        return render_template('index.html',value='It is a bad transaction')
    else:
        return render_template('index.html',value='It is a good transaction')


if __name__=='main':
    app.run(debug=True)