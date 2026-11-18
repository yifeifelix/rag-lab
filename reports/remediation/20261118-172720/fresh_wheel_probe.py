from pathlib import Path
import json,os,shutil,subprocess,sys,tempfile,zipfile
root=Path.cwd(); audit=Path((root/'.cache/remediation_dir').read_text().strip())
base=Path(tempfile.mkdtemp(prefix='rag-image-',dir=root/'.cache/tmp'));app=base/'app';app.mkdir();site=base/'site'
with zipfile.ZipFile(next((root/'dist').glob('*.whl'))) as archive:archive.extractall(site)
for folder in ['learning_v2','corpus','eval']:
 shutil.copytree(root/'data'/folder,app/'data'/folder)
(app/'scripts').mkdir()
shutil.copyfile(root/'scripts/run_learning_study.py',app/'scripts/run_learning_study.py')
env={key:value for key,value in os.environ.items() if not key.startswith('RAG_')}
env.update(RAG_PROJECT_ROOT=str(app),RAG_DATA_ROOT=str(app),RAG_LLM_MODE='local',RAG_EMBEDDING_BACKEND='local_svd',
 RAG_ADMIN_TOKEN='acceptance-owner-value',RAG_TESTER_TOKEN='acceptance-tester-value',
 RAG_REPORTS_DIR=str(app/'reports/private'),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(site))
command=[sys.executable,str(app/'scripts/run_learning_study.py'),'calibrate','--output',str(app/'reports/private/calibration.json')]
result=subprocess.run(command,env=env,capture_output=True,text=True);(audit/'fresh_wheel_calibration.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stderr
env['RAG_CALIBRATION_PATH']=str(app/'reports/private/calibration.json')
code=r'''
from pathlib import Path
import json,os
import rag_platform,rag_platform.server
from fastapi.testclient import TestClient
assert str(Path(rag_platform.server.__file__)).startswith(os.environ['PYTHONPATH'])
app=rag_platform.server.create_app()
with TestClient(app) as client:
 health=client.get('/health').json()
 assert health['status']=='healthy' and health['corpus_documents']==81 and health['calibration_loaded'],health
 assert client.get('/').status_code==200
 assert client.get('/api/v1/documents').status_code==401
 tester={'Authorization':'Bearer acceptance-tester-value'}
 owner={'Authorization':'Bearer acceptance-owner-value'}
 assert client.get('/api/v1/capabilities',headers=tester).json()['role']=='tester'
 assert client.post('/api/v1/documents',headers=tester,json={}).status_code==403
 response=client.post('/api/v1/query',headers=tester,json={'query':'What is the current EU restore recovery target for Cedar?'})
 assert response.status_code==200,response.text
 payload=response.json()
 assert payload['confidence_kind']=='dev_calibrated_answerability'
 assert payload['request_id']==response.headers['x-request-id']
 assert 'compression_ratio' not in payload['stage_timings_ms']
 stream=client.post('/api/v1/query/stream',headers=tester,json={'query':'What is the current EU restore recovery target for Cedar?'})
 assert stream.status_code==200 and stream.headers['x-streaming-mode']=='buffered-response-replay'
 block=stream.text.split('\n\n')[0]
 metadata=json.loads(next(line[6:] for line in block.splitlines() if line.startswith('data: ')))
 assert metadata['request_id'] and 'evidence' in metadata
 feedback=client.post('/api/v1/feedback',headers=tester,json={'request_id':payload['request_id'],'rating':3,'issue':'other','comment':'Automated acceptance for synthetic workspace, not a human quality rating.'})
 assert feedback.status_code==200
 doc={'doc_id':'acceptance.doc','title':'Acceptance fixture','domain':'support','text':'Order 1234567890123; telephone +1 415 555 0100 2345; test card 4111 1111 1111 1111.'}
 assert client.post('/api/v1/documents',headers=owner,json=doc).status_code==200
 assert client.get('/api/v1/documents/acceptance.doc',headers=owner).json()['document']['text']==doc['text']
 assert client.delete('/api/v1/documents/acceptance.doc',headers=owner).status_code==200
 result={'status':'passed','version':rag_platform.__version__,'transport':'ASGI in memory; no Docker or socket',
 'isolated_wheel_import':True,'initial_documents':health['corpus_documents'],'calibration_loaded':health['calibration_loaded'],
 'checks':['fresh workspace bootstrap','UI','health','401 authentication','tester role','403 mutation protection','query response','buffered SSE evidence','feedback persistence','owner source preservation','owner delete'],
 'query_observation':{'answer':payload['answer'],'refused':payload['refused'],'confidence':payload['confidence'],'citations':payload['citations']},'human_quality_review':'pending'}
Path(os.environ['ACCEPTANCE_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
print('Fresh wheel acceptance passed')
'''
env['ACCEPTANCE_RESULT']=str(audit/'fresh_wheel_acceptance.json')
result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
(audit/'fresh_wheel_acceptance.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
print(result.stdout.strip());print('Evidence:',audit/'fresh_wheel_acceptance.json')
