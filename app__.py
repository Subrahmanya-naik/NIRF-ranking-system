import numpy as np
from flask import Flask, request,  render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('F:\python_flask\MODEL_.pkl', 'rb'))
poly = pickle.load(open('F:\python_flask\POLYNOMIALTRANSFORM_.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexfinal.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    tlr = float(request.form['TLR'])
    rpc = float(request.form['RPC'])
    go = float(request.form['GO'])
    oi =float(request.form['OI'])
    ppn =float(request.form['PPN'])


    tlr=(tlr)/(94.4)
    rpc=(rpc)/(96.93)
    go=(go)/(90.605)
    oi=(oi)/(68.5)
    ppn=(ppn)/(100.0)

    score=(tlr+rpc)*0.3+go*0.2+(oi+ppn)*0.1

   
    dif=[6, 4, 3, 9, 12, 10, 6, 2, 2, 3, 6, 8, 4, 3, 4, 6, 6]
    rank= model.predict(poly.fit_transform([[score]]))
    if rank < 0:
        rank=1
    if rank>169:
        output="Rank is above 169"
    else:
        er=int(rank//10)
        output=str(int(rank))+'±'+str(dif[er])


    return render_template('indexfinal.html', prediction_text='Rank of the college will be: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)