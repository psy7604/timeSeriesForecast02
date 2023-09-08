from flask import Flask, request, render_template
import process
# from flask_cors import CORS

app = Flask(__name__, static_folder='./dist', template_folder='./dist', static_url_path='')
# CORS(app)


@app.route('/')
def index():
    return render_template('index.html', name='index')


@app.route('/static')
def work1():
    serverID = int(request.args.get('serverID'))
    period = int(request.args.get('period'))
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')

    results = process.staticForecast(serverID, period, startDate, endDate)
    return results


@app.route('/dynamic')
def work2():
    serverID = int(request.args.get('serverID'))
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')

    results = process.dynamicForecast(serverID, startDate, endDate)
    return results


if __name__ == '__main__':
    app.run(port=8090)
