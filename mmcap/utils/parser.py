"""
Luke
"""
import numpy as np


def weather_parser(data:dict) -> list:
    """Parse json weather
----------------------------------------
Category    항목명          단위 
----------------------------------------
POP	    강수확률        %
PTY	    강수형태        코드값
R06	    6시간 강수량    범주 (1 mm)
REH	    습도            %
S06	    6시간 신적설    범주(1 cm)
SKY	    하늘상태        코드값
T3H	    3시간 기온      ℃
TMN	    아침 최저기온   ℃
TMX	    낮 최고기온     ℃
UUU	    풍속(동서성분)  m/s
VVV	    풍속(남북성분)  m/s
WAV	    파고            M
VEC	    풍향            m/s
WSD	    풍속            m/s
----------------------------------------
일단 요거만 씀
- 하늘상태(SKY) 코드 : 맑음(1), 구름많음(3), 흐림(4) 

    """

    header = data['response']['header']
    body = data['response']['body']

    # parse header - code, msg
    result_code = header['resultCode']
    result_msg = header['resultMsg']

    # parse body - data_type, items, page_no, num_of_rows, total_count
    data_type = body['dataType']
    items = body['items']
    page_no = body['pageNo']
    num_of_rows = body['numOfRows']
    total_count = body['totalCount']

    item_list = items['item'] # length 10

    result_list = list()

    for item in item_list:
        # item - base_date, base_time, category, fcst_date, fcst_time, fcst_value, nx, ny
        base_date = item['baseDate']
        base_time = item['baseTime']
        category = item['category']
        fcst_date = item['fcstDate']
        fcst_time = item['fcstTime']
        fcst_value = item['fcstValue']
        nx = item['nx']
        ny = item['ny']
        # FIXME : 일단 "SKY" category 만 받아옴 
        if category == "SKY":
            result_list.append(item)

    return result_list


def context_parser(context:dict, model_type:str):
    """Context Parser
    Generate input TURN by chatbot model_type

    """
    if model_type == 'HQA':
        return parse_HQA(context)
    elif model_type == 'QA_ter':
        return parse_QA_ter(context)


def parse_HQA(context:dict):
    utterance_list = [dic['utterance'] for dic in context]
    question = utterance_list[-1]

    try:
        h3 = utterance_list[-2]
    except:
        h3 = np.nan

    try:
        h2 = utterance_list[-3]
    except:
        h2 = np.nan
    
    try:
        h1 = utterance_list[-4]
    except:
        h1 = np.nan

    turn = dict(h1=h1,
                h2=h2,
                h3=h3,
                Q=question,
                A='')

    return turn


def parse_QA_ter(context:dict):
    utterance_list = [dic['utterance'] for dic in context]
    question = utterance_list[-1]
    predicted_next_ai_emotion = context[-1]['next_ai_emotion']

    turn = dict(Q=question,
                A='',
                TER=predicted_next_ai_emotion)

    return turn 
