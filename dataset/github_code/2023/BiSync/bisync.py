import logging
import pyonmttok
import ctranslate2
import edit_distance
from flask import Flask, request, json, jsonify
from flask_cors import CORS

### run this script on the server:
# export FLASK_APP=bisync.py
# flask run --host=0.0.0.0 --port=5000
# or
# flask run --host=127.0.0.1 --port=5000
### to test, you can send client data using:
#curl -X POST -H "Content-type: application/json" -d "{\"src\": \"This is my new sentence.\", \"lang\":\"｟fr｠\", \"tgt\":\"\", \"mode\":\"sync\"}" "http://127.0.0.1:5000/"

logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None), filename='./bisync.log')

mdir = "./ct2_en-fr_int8"
translator = ctranslate2.Translator(mdir, device="cpu")
logging.info('loaded Model {}'.format(mdir))
onmttok = pyonmttok.Tokenizer('aggressive', joiner_annotate=True, segment_numbers=True, bpe_model_path=mdir+"/bpe_32k")
onmttok_noannotate = pyonmttok.Tokenizer('aggressive', joiner_annotate=False, segment_numbers=True, bpe_model_path=mdir+"/bpe_32k")
logging.info('loaded Tokenizer')
app = Flask(__name__)
CORS(app)
@app.route('/', methods=['POST'])


def sync_request():
    req = request.json
    logging.info('REQUEST: {}'.format(req))
    oraw = translate(req['src'], req['tgt'], req['lang'], req['tgt_ini'], req['src_ini'], req['mode'], int(req['alt']))
    res = {"oraw" : oraw}
    logging.info('RESPONSE: {}'.format(res))
    return jsonify(res)


def update_type(src, src_ini):
    if len(src_ini) == 0:
        return 'scratch'
    #using onmttok_noannotate is to avoid differences in spaces (does not annotate joiner)
    sm = edit_distance.SequenceMatcher(a=onmttok_noannotate(src_ini), b=onmttok_noannotate(src), action_function=edit_distance.highest_match_action)
    nins, ndel, nsub = 0, 0, 0
    for (code, b1, e1, b2, e2) in sm.get_opcodes():
        if code == 'insert':
            nins +=1
        elif code == 'delete':
            ndel +=1
        elif code == 'replace':
            nsub +=1
    logging.info('ins:{} del:{} sub:{}'.format(nins,ndel,nsub))
    if nins+ndel+nsub > 9: #too many changes imply translate from scratch
        return 'scratch'
    if   nins > 0 and ndel == 0 and nsub == 0:
        return '｟INS｠'
    elif nins == 0 and ndel > 0 and nsub == 0:
        return '｟DEL｠'
    elif nins >= 0 and ndel == 0 and nsub > 0:
        return '｟SUB｠'
    elif nins == 0 and ndel >= 0 and nsub > 0:
        return '｟SUB｠'
    logging.info('A:{}'.format(onmttok_noannotate(src_ini)))
    logging.info('B:{}'.format(onmttok_noannotate(src)))
    return 'scratch'


def translate(src, tgt, lang, tgt_ini, src_ini, mode, alt):
    logging.info('ini: {} || {}'.format(src_ini, tgt_ini))
    logging.info('cur: {} || {}'.format(src, tgt))

    if mode == 'sync':
        if len(tgt) == 0: #a translation from scratch
            iraw = src + ' ' + lang
        else:
            update = update_type(src, src_ini)
            if update == 'scratch': #finally, a translation from scratch
                iraw = src + ' ' + lang
            else:
                iraw = src + ' ' + lang + ' ' + tgt + ' ' + update
        itok = onmttok(iraw)
        results = translator.translate_batch([itok], disable_unk=True)
        otok = results[0].hypotheses
        oraw = [onmttok.detokenize(t) for t in otok]

    elif mode == 'gap':
        iraw = src + ' ' + lang + ' ' + tgt
        itok = onmttok(iraw)
        results = translator.translate_batch([itok], beam_size=alt, num_hypotheses=alt, return_alternatives=True, disable_unk=True)
        otok = results[0].hypotheses
        oraw = [onmttok.detokenize(t) for t in otok]

    elif mode == 'pref':
        iraw = src + ' ' + lang 
        itok = onmttok(iraw)
        praw = tgt
        ptok = onmttok(praw)
        results = translator.translate_batch([itok], target_prefix=[ptok], beam_size=alt, num_hypotheses=alt, return_alternatives=True, disable_unk=True)
        otok = results[0].hypotheses
        oraw = [onmttok.detokenize(t[len(ptok):]) for t in otok]

    logging.info('iraw: {}'.format(iraw))
    logging.info('oraw: {}'.format(oraw[0]))
    return oraw #list with alternatives
