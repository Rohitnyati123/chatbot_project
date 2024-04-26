from flask import Flask, redirect, url_for, render_template, request 
from src.pipeline.prediction import generate_response
from src.pipeline.prediction import train_model
import pickle

"""this is the user window"""
app=Flask(__name__)
@app.route('/')
def demo():
    # input=request.form.get('inputText')
    return render_template('index.html')


"""this is the result window"""
@app.route('/success',methods=['POST'])
def printdata():
    with open('model2.pkl', 'rb') as f:
        model = pickle.load(f)
    # model,length=train_model()
    while True:
        result=request.form.get('inputText')
        # if result.lower() == 'quit':
        #     break
        response=[]
        response.append(generate_response(result,model=model))
        if response is None:
            return f" no output {response}"
        else: 
            return render_template('result.html',result=response) 


if __name__=='__main__':
    app.run(debug=True)