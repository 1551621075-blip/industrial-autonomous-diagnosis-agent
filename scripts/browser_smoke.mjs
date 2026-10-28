#!/usr/bin/env node
/** Actual installed macOS Chrome + Node's built-in WebSocket; no npm/browser downloads.
 * node scripts/browser_smoke.mjs --java /explicit/local/jdk/bin/java
 * Alternatively supply INDUSTRIAL_DIAGNOSIS_JAVA, JAVA_PATH, or JAVA_HOME explicitly.
 * Original synthetic telemetry, real HermiT, exact SyntheticProvider; NOT LLM acceptance.
 * Creates only an owned temp app/profile. No real profiles/cookies, config, or model endpoints.
 * Setup credentials travel on child stdout piped to this parent and remain in memory.
 * No token command arguments, environment, URLs, screenshots, output, or credential files.
 */
import {spawn, execFile} from 'node:child_process';
import {constants} from 'node:fs';
import {request as httpRequest} from 'node:http';
import {access, mkdtemp, readFile, readdir, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, isAbsolute, join, resolve} from 'node:path';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual, promisify} from 'node:util';

const PROJECT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PYTHON = join(PROJECT, '.venv/bin/python');
const START = Date.now();
const results = [], children = [], secrets = [], responses = [], faults = [];
const requests = new Map(), controls = new Map(), external = new Set();
let root, server, chrome, cdp, sessionId, bootstrap, origin, stopping = false, sequence = 0;
let normal, jam, report, unknown, unattested;
const XSS = '<img id="r3-smoke-literal" src="/r3-smoke-never-image" onerror="window.__r3SmokeXSS=73">';
const COMPLAINT = `ORIGINAL SYNTHETIC browser investigation; literal text only: ${XSS}`;
const OBSERVATION = `ORIGINAL SYNTHETIC test observation: obstruction_confirmed=true. No physical inspection was performed. ${XSS}`;
const REVIEW = `ORIGINAL SYNTHETIC browser review of exact report content; no actuation or physical validation. ${XSS}`;
const CLEAR_INPUTS = ['token', 'complaint', 'observation', 'review-note', 'signoff-note', 'call-id', 'reconcile-note', 'reconcile-tokens', 'knowledge-file'];
const UI_CASES = [
  'anonymous_and_TEST_mode', 'wrong_token_DOM_error', 'real_identity_cases_tasks_and_memory_only_token',
  'real_HermiT_checked_version_and_explicit_UI_publication', 'normal_UI_submit_and_original_evidence',
  'normal_report_exact_content_and_untrusted_TEST_draft', 'normal_explicit_immutable_signoff',
  'literal_DOM_injection_is_text', 'task_switch_clears_previous_report_and_audit',
  'jam_pauses_without_inventing_human_evidence',
  'jam_explicit_synthetic_human_answer_then_submit', 'real_audit_and_no_fixture_oracle_in_requests',
  'unknown_result_no_automatic_retry_and_guarded_controls', 'unknown_explicit_owner_admin_reconciliation',
  'unknown_blank_usage_conservatively_exhausts_budget',
  'operator_role_controls_and_owner_isolation', 'invalid_replacement_login_clears_old_identity',
  'logout_clears_token_and_sensitive_DOM', 'logout_rejects_late_real_response',
  'reload_forgets_identity', 'security_headers_on_success_and_boundary_errors',
  'no_external_requests_console_tokens_or_persistent_credentials',
];
function assert(value, message) { if (!value) throw new Error(message); }
function safe(value) {
  let text = String(value);
  for (const secret of secrets) text = text.split(secret).join('[REDACTED]');
  return text.slice(0, 1200);
}
function record(name, status, detail) { results.push({name, status, ...(detail ? {detail: safe(detail)} : {})}); }
async function test(name, work, prerequisite = true) {
  if (stopping) return false;
  if (!prerequisite) { record(name, 'not_tested', 'Prerequisite failed; no API/fixture bypass of the UI flow'); return false; }
  try { await work(); record(name, 'pass'); return true; }
  catch (error) { record(name, 'fail', error.message); return false; }
}
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function deadline(promise, ms, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label)), ms);
  })]); } finally { clearTimeout(timer); }
}
async function until(work, label, timeout = 12000) {
  const end = Math.min(Date.now() + timeout, START + 270000);
  while (!stopping && Date.now() < end) {
    const value = await work();
    if (value) return value;
    await delay(35);
  }
  throw new Error(`Timed out: ${label}`);
}
function javaArgument() {
  const args = process.argv.slice(2);
  assert(args.length === 0 || (args.length === 2 && args[0] === '--java'), 'Usage: node scripts/browser_smoke.mjs --java /absolute/jdk/bin/java');
  const path = args[1] || process.env.INDUSTRIAL_DIAGNOSIS_JAVA || process.env.JAVA_PATH ||
    (process.env.JAVA_HOME ? join(process.env.JAVA_HOME, 'bin/java') : null);
  assert(path && isAbsolute(path), 'Real HermiT requires explicit absolute --java, INDUSTRIAL_DIAGNOSIS_JAVA, JAVA_PATH, or JAVA_HOME; no reasoner fallback');
  return path;
}
function environment(java) {
  return {PATH: `${join(PROJECT, '.venv/bin')}:/usr/bin:/bin:/usr/sbin:/sbin`, HOME: root,
    TMPDIR: root, LANG: 'en_US.UTF-8', PYTHONPATH: join(PROJECT, 'src'),
    PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1',
    ...(java ? {INDUSTRIAL_DIAGNOSIS_JAVA: java} : {})};
}
function launch(executable, args, java, stdio) {
  assert(!stopping, 'Acceptance is stopping; no new child may start');
  const child = spawn(executable, args, {cwd: PROJECT, env: environment(java), detached: true, stdio});
  children.push(child);
  child.on('error', () => { child.launchFailed = true; });
  // Diagnostics are discarded, never forwarded or written to a log.
  child.stderr?.resume();
  return child;
}

const PY_SERVER = String.raw`
import asyncio, json, socket, sys, threading
from pathlib import Path
import uvicorn
from industrial_diagnosis.api import create_app
from industrial_diagnosis.fixtures import original_knowledge
from industrial_diagnosis.provider import SyntheticProvider
from industrial_diagnosis.schemas import DomainError

channel = sys.stdout
sys.stdout = sys.stderr  # only the explicit protocol may write to the parent's pipe
lock = threading.Lock()
def send(value):
    with lock:
        channel.write(json.dumps(value, ensure_ascii=True) + '\n')
        channel.flush()

try:
    provider = SyntheticProvider()
    app = create_app(Path(sys.argv[1]), provider=provider, java_path=Path(sys.argv[2]))
    store = app.state.store
    tokens = {role: store.provision_user('browser-' + role, role, ['PLANT-A'])
              for role in ('admin', 'operator')}
    actor = store.authenticate(tokens['admin'])
    listing = app.state.telemetry.seed('PLANT-A')
    built = app.state.knowledge.build(original_knowledge(), actor)
    if built['status'] != 'checked':
        send({'type':'failed', 'code':'real_hermit_' + built['reasoner']['status']})
        raise SystemExit(1)
    # Explicit synthetic PRE-DECISION timeout. The service treats the outcome as
    # unknown, but this fixture can attest zero because it throws before calling the
    # deterministic provider. No model inference or database/checkpoint fabrication.
    fault = {'armed': False, 'calls': 0, 'decisions': 0, 'injected': 0}
    original_decision = provider.decision
    def decision(context):
        fault['calls'] += 1
        if fault['armed']:
            fault['armed'] = False
            fault['injected'] += 1
            raise DomainError('provider_unknown', 'Explicit synthetic pre-decision timeout browser fixture', 503)
        fault['decisions'] += 1
        return original_decision(context)
    provider.decision = decision
    # Delay a REAL app response, not a forged payload, only for the logout race test.
    gate = {'armed': False, 'held': False}
    released = threading.Event()
    class DelayedResponse:
        async def __call__(self, scope, receive, reply):
            hold = (scope['type'] == 'http' and scope['method'] == 'GET'
                    and scope['path'] == '/api/cases' and gate['armed'])
            if hold:
                gate['armed'] = False
            async def send_response(message):
                if hold and message['type'] == 'http.response.start':
                    gate['held'] = True
                    await asyncio.to_thread(released.wait, 15)
                    gate['held'] = False
                await reply(message)
            await app(scope, receive, send_response)
    def control():
        for line in sys.stdin:
            msg = {}
            try:
                msg = json.loads(line)
                command = msg['command']
                if command == 'hold_cases':
                    released.clear()
                    gate['armed'] = True
                    value = {'armed': True}
                elif command == 'release_cases':
                    released.set()
                    value = {'released': True}
                elif command == 'unknown_once':
                    fault['armed'] = True
                    value = {'armed': True}
                elif command == 'state':
                    value = {'held': gate['held'], 'provider_calls': fault['calls'],
                             'synthetic_decisions': fault['decisions'], 'injected_timeouts': fault['injected']}
                else:
                    raise ValueError('Unsupported control')
                send({'id': msg['id'], 'value': value})
            except Exception:
                send({'id': msg.get('id'), 'error': 'fixture_control_failed'})
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    threading.Thread(target=control, daemon=True).start()
    send({'type':'ready', 'port':sock.getsockname()[1], 'tokens':tokens,
          'cases':listing['cases'], 'knowledge':built})
    uvicorn.Server(uvicorn.Config(DelayedResponse(), host='127.0.0.1',
        port=sock.getsockname()[1], log_level='critical', access_log=False,
        log_config=None, server_header=False)).run(sockets=[sock])
except Exception as exc:
    # Never include exception text/traceback, request content, or credentials in protocol errors.
    send({'type':'failed', 'code':getattr(exc, 'code', type(exc).__name__)})
    raise SystemExit(1) from None
`;

class CDP {
  constructor(socket) {
    this.socket = socket; this.next = 0; this.pending = new Map(); this.listeners = new Set();
    socket.addEventListener('message', event => {
      let value;
      try { value = JSON.parse(event.data); } catch { this.close(); return; }
      const pending = this.pending.get(value.id);
      if (pending) {
        clearTimeout(pending.timer); this.pending.delete(value.id);
        if (value.error) pending.reject(new Error(`CDP ${pending.method} failed (${value.error.code})`));
        else pending.resolve(value.result);
      } else for (const handler of this.listeners) Promise.resolve(handler(value)).catch(() => faults.push('CDP observer failed'));
    });
    socket.addEventListener('close', () => this.close());
    socket.addEventListener('error', () => this.close());
  }
  send(method, params = {}, sessionId) {
    assert(!stopping && this.socket.readyState === WebSocket.OPEN, 'CDP connection unavailable');
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`CDP ${method} timeout`)); }, 15000);
      this.pending.set(id, {resolve, reject, timer, method});
      this.socket.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
    });
  }
  close() {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer); pending.reject(new Error(`CDP closed during ${pending.method}`));
    }
    this.pending.clear();
    if (this.socket.readyState === WebSocket.OPEN) this.socket.close();
  }
}
const cmd = (method, params = {}) => cdp.send(method, params, sessionId);
const lit = value => JSON.stringify(value);
async function evaluate(expression) {
  const value = await cmd('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  assert(!value.exceptionDetails, 'Browser evaluation threw (payload withheld)');
  return value.result?.value;
}
async function click(selector) {
  const point = await evaluate(`(() => {
    const e = document.querySelector(${lit(selector)});
    if (!e || e.disabled || e.closest('[hidden]')) return null;
    e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect();
    return r.width && r.height ? {x:r.x+r.width/2,y:r.y+r.height/2} : null;
  })()`);
  assert(point, `Not a visible enabled click target: ${selector}`);
  await cmd('Input.dispatchMouseEvent', {type:'mousePressed', button:'left', clickCount:1, ...point});
  await cmd('Input.dispatchMouseEvent', {type:'mouseReleased', button:'left', clickCount:1, ...point});
}
async function fill(selector, value) {
  assert(await evaluate(`(() => {const e=document.querySelector(${lit(selector)});
    if(!e || e.disabled || e.closest('[hidden]')) return false;
    e.scrollIntoView({block:'center'}); e.focus(); e.select(); return true;})()`), `Missing input ${selector}`);
  await cmd('Input.insertText', {text:String(value)});
}
async function choose(selector, value) {
  // Real DOM selection/change event invokes the shipped listener; no application functions called.
  assert(await evaluate(`(() => {const e=document.querySelector(${lit(selector)});
    if(!e || e.disabled || ![...e.options].some(o=>o.value===${lit(value)})) return false;
    e.value=${lit(value)}; e.dispatchEvent(new Event('change',{bubbles:true})); return true;})()`), `Missing option in ${selector}`);
  await idle();
}
async function idle() { await until(() => evaluate('typeof busy === "boolean" && !busy'), 'UI handler idle'); }
async function responseAfter(mark, path, method = 'POST') {
  return until(() => responses.slice(mark).find(r => r.path === path && r.method === method && r.done), `${method} ${path}`, 45000);
}
async function submit(selector, path, method = 'POST') {
  const mark = responses.length;
  await click(selector);
  const response = await responseAfter(mark, path, method);
  await idle();
  return response;
}
function ok(response, label) {
  assert(response.status === 200 && response.body, `${label}: HTTP ${response.status}; code=${response.body?.error || 'none'}`);
  return response.body;
}
async function control(command) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { controls.delete(id); reject(new Error('Fixture control timeout')); }, 5000);
    controls.set(id, {resolve, reject, timer});
    server.stdin.write(JSON.stringify({id, command}) + '\n');
  });
}
async function browserProbe(path, role, method = 'GET', body, extraHeaders = {}) {
  // Boundary/replay probes have no separate form. They still use actual Chrome-origin HTTP.
  const token = role ? bootstrap.tokens[role] : '';
  return evaluate(`(async () => {const r=await fetch(${lit(path)}, {method:${lit(method)},
    credentials:'omit', cache:'no-store', headers:${lit({...(token ? {Authorization:'Bearer '+token} : {}),
      ...(body === undefined ? {} : {'Content-Type':'application/json'}), ...extraHeaders})},
    ${body === undefined ? '' : `body:${lit(JSON.stringify(body))},`}});
    return {status:r.status,body:await r.json(),headers:Object.fromEntries(r.headers)};})()`);
}
async function login(role) {
  await fill('#token', bootstrap.tokens[role]);
  ok(await submit('#login button:not(#logout)', '/api/me', 'GET'), `${role} login`);
  assert(await evaluate(`document.getElementById('identity').textContent.includes(${lit('browser-'+role)}) && document.getElementById('token').value===''`), 'Identity/token input not updated');
}
async function selectCase(name) {
  const row = bootstrap.cases[name];
  await choose('#cases', `${row.event_id}:${row.device_id}:${row.plant_id}`);
}
async function newTask(name, complaint) {
  await selectCase(name);
  if (complaint) await fill('#complaint', complaint);
  const mark = responses.length;
  const created = ok(await submit('#new-task button', '/api/tasks'), 'Task creation');
  const run = ok(await responseAfter(mark, `/api/tasks/${created.task_id}/run`), 'Task run');
  await idle();
  return run;
}
async function openTask(task) {
  await submit('#refresh-tasks', '/api/tasks', 'GET');
  const mark = responses.length;
  await choose('#tasks', task.task_id);
  ok(await responseAfter(mark, `/api/tasks/${task.task_id}`, 'GET'), 'Open task');
}
async function taskDOM(task) {
  const view = await evaluate(`({status:document.getElementById('task-status').textContent,
    rows:[...document.querySelectorAll('#evidence tr')].map(r=>[...r.children].map(c=>c.textContent)),
    submission:JSON.parse(document.getElementById('submission').textContent)})`);
  assert(view.status.includes(task.status), 'Task status not rendered');
  assert(isDeepStrictEqual(view.submission, task.submission), 'Rendered frozen submission differs from API');
  assert(view.rows.length === task.evidence.length, 'Evidence row count differs from actual tool output');
  task.evidence.forEach((e,i) => assert(isDeepStrictEqual(view.rows[i],
    [e.condition_id,e.state,e.source,JSON.stringify(e.details),e.evidence_id]), 'Evidence DOM differs from API'));
  assert(task.evidence.every(e=>e.evidence_id && e.details && Object.keys(e.details).length), 'Evidence lacks real tool details');
}
async function reconciliationAudit(task, usage, conservative, note) {
  if(!await evaluate(`document.querySelector('details:has(#audit)').open`))
    await click('details:has(#audit) > summary');
  const audit=ok(await submit('#refresh-audit',`/api/tasks/${task.task_id}/audit`,'GET'),'Reconciliation audit');
  assert(isDeepStrictEqual(await evaluate(`JSON.parse(document.getElementById('audit').textContent)`),audit),
    'Reconciliation audit DOM differs from actual API');
  const calls=audit.model_calls.filter(row=>row.call_id===task.pending_call_id);
  const events=audit.events.filter(row=>row.event==='model_reconciled' &&
    JSON.parse(row.detail_json).call_id===task.pending_call_id);
  assert(calls.length===1 && calls[0].status==='confirmed_no_result' && calls[0].tokens===usage && events.length===1,
    'Reconciled call usage/status or unique audit event missing');
  const detail=JSON.parse(events[0].detail_json);
  assert(events[0].actor_id==='browser-admin' && detail.no_result===true && detail.tokens===usage &&
    detail.conservative_usage===conservative && detail.note===note && detail.human_attestation_not_external_proof===true,
    'Reconciliation audit lost exact usage, synthetic attestation, or conservative accounting provenance');
}
async function xssCheck() {
  assert(await evaluate(`!window.__r3SmokeXSS && !document.getElementById('r3-smoke-literal') && !document.querySelector('#submission img,#report img,#evidence img,#audit img')`), 'Literal input became HTML or executed');
  assert(!requests.size || ![...requests.values()].some(r=>r.path==='/r3-smoke-never-image'), 'Literal image triggered HTTP');
}
async function storageCheck() {
  const value = await evaluate(`(async()=>({local:Object.entries(localStorage),session:Object.entries(sessionStorage),
    cookie:document.cookie,databases:await indexedDB.databases(),caches:await caches.keys(),
    dom:document.documentElement.outerHTML,inputs:[...document.querySelectorAll('input,textarea')].map(e=>e.value),url:location.href}))()`);
  const cookies = (await cdp.send('Storage.getCookies')).cookies;
  assert(!secrets.some(s=>JSON.stringify({value,cookies}).includes(s)), 'Credential leaked into DOM, URL, or browser storage');
  assert(!value.local.length && !value.session.length && !value.cookie && !value.databases.length &&
    !value.caches.length && !cookies.length, 'Fresh browser acquired persistent storage/cookies');
}
async function cleared() {
  const state = await evaluate(`({tokenEmpty:typeof token==='string' && token==='',
    taskEmpty:currentTask===null,reportEmpty:currentReport===null,
    fields:${lit(CLEAR_INPUTS)}.map(id=>({id,value:document.getElementById(id)?.value})),
    options:['cases','versions','tasks','reports'].map(id=>[...document.getElementById(id).options].some(o=>o.value)),
    humanHidden:document.getElementById('human').hidden,question:document.getElementById('question').textContent,
    predicate:document.getElementById('human-predicate').textContent,answer:document.getElementById('answer-value').value,
    rows:document.querySelectorAll('#evidence tr').length,text:document.body.textContent})`);
  assert(state.tokenEmpty && state.taskEmpty && state.reportEmpty, 'Token/current objects survived identity clearing');
  const dirty = state.fields.filter(f=>f.value);
  assert(!dirty.length, `Sensitive input(s) survived clearing: ${dirty.map(f=>f.id).join(', ')}`);
  assert(!state.options.some(Boolean) && state.humanHidden && !state.answer &&
    [state.question,state.predicate].every(value=>!value || value==='已清除本地身份及页面数据。') && !state.rows,
  'Old choices/human question/predicate/answer/evidence survived clearing');
  for (const id of [normal?.task_id,jam?.task_id,report?.report_id,unknown?.task_id,unattested?.task_id,'browser-admin','browser-operator'])
    if (id) assert(!state.text.includes(id), 'Previous identity or resource remains in page text');
}
async function disabled(selector) {
  return evaluate(`(() => {const e=document.querySelector(${lit(selector)}); return !!e && (e.disabled || !!e.closest('[hidden]') || getComputedStyle(e).display==='none');})()`);
}
function headersCheck(headers, label) {
  const h = Object.fromEntries(Object.entries(headers).map(([k,v])=>[k.toLowerCase(),String(v)]));
  assert(h['cache-control']?.includes('no-store') && h['x-content-type-options']==='nosniff' &&
    h['referrer-policy']==='no-referrer' && h['x-frame-options']==='DENY', `${label}: missing no-store/nosniff/no-referrer/DENY`);
  const csp = h['content-security-policy'] || '';
  assert(["default-src 'self'","script-src 'self'","object-src 'none'","frame-ancestors 'none'","base-uri 'none'"].every(s=>csp.includes(s)) &&
    !/unsafe-inline|unsafe-eval/.test(csp), `${label}: missing restrictive CSP`);
}

async function start(java) {
  assert(process.platform === 'darwin', 'Acceptance requires installed macOS Chrome');
  assert(typeof WebSocket === 'function', 'Node built-in WebSocket required (Node 22+)');
  await Promise.all([CHROME,PYTHON,java].map(path=>access(path, constants.X_OK)));
  root = await mkdtemp(join(tmpdir(), 'r3-original-browser-'));
  server = launch(PYTHON, ['-u','-c',PY_SERVER,join(root,'app'),java], java, ['pipe','pipe','ignore']);
  bootstrap = await deadline(new Promise((resolve,reject) => {
    createInterface({input:server.stdout}).on('line', line => {
      let value;
      try { value=JSON.parse(line); } catch { reject(new Error('Invalid setup protocol (content withheld)')); return; }
      if(value.type==='ready') {
        secrets.push(...Object.values(value.tokens)); resolve(value); return;
      }
      if(value.type==='failed') { reject(new Error(`Synthetic app setup failed: ${value.code}`)); return; }
      const pending=controls.get(value.id);
      if(pending) { clearTimeout(pending.timer); controls.delete(value.id);
        if(value.error) pending.reject(new Error(value.error)); else pending.resolve(value.value); }
    });
    server.once('exit',()=>reject(new Error('Synthetic app exited before readiness')));
    server.once('error',()=>reject(new Error('Project Python launch failed')));
  }), 90000, 'Real HermiT/setup timed out');
  origin = `http://127.0.0.1:${bootstrap.port}`;
  const profile=join(root,'profile');
  chrome = launch(CHROME, ['--headless=new','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
    `--user-data-dir=${profile}`, '--no-first-run','--no-default-browser-check',
    '--disable-background-networking','--disable-component-update','--disable-default-apps','--disable-sync',
    '--disable-breakpad','--disable-crash-reporter','--disable-domain-reliability','--safebrowsing-disable-auto-update',
    '--disable-client-side-phishing-detection','--metrics-recording-only','--password-store=basic','--use-mock-keychain',
    '--no-proxy-server','--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1',
    '--disable-features=AutofillServerCommunication,OptimizationHints,MediaRouter,Translate,PasswordLeakDetection',
    '--window-size=1440,1100','about:blank'], null, ['ignore','ignore','ignore']);
  const active = await until(async()=>{
    assert(chrome.exitCode===null && !chrome.launchFailed, 'Installed Chrome exited before CDP readiness');
    try { const lines=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).trim().split('\n');
      return /^\d+$/.test(lines[0]) && lines[1]?.startsWith('/devtools/browser/') ? lines : null;
    } catch { return null; }
  }, 'Owned Chrome DevToolsActivePort', 15000);
  const socket = new WebSocket(`ws://127.0.0.1:${active[0]}${active[1]}`);
  await deadline(new Promise((resolve,reject)=>{
    socket.addEventListener('open',resolve,{once:true});
    socket.addEventListener('error',()=>reject(new Error('Chrome WebSocket failed')),{once:true});
  }),10000,'Chrome WebSocket timeout');
  cdp = new CDP(socket);
  const version=await cdp.send('Browser.getVersion');
  record('actual_installed_Chrome_builtin_WebSocket', 'pass', `${version.product}; ${version.userAgent}; JS ${version.jsVersion}`);
  const target=await cdp.send('Target.createTarget',{url:'about:blank'});
  sessionId=(await cdp.send('Target.attachToTarget',{targetId:target.targetId,flatten:true})).sessionId;
  cdp.listeners.add(async event => {
    if(event.sessionId!==sessionId) return;
    const p=event.params;
    if(event.method==='Runtime.exceptionThrown') faults.push('Uncaught browser exception');
    if(event.method==='Runtime.consoleAPICalled' && secrets.some(s=>JSON.stringify(p.args).includes(s))) faults.push('Credential in browser console');
    if(event.method==='Page.javascriptDialogOpening') {
      faults.push('Unexpected JavaScript dialog'); await cmd('Page.handleJavaScriptDialog',{accept:false});
    }
    if(event.method==='Network.requestWillBeSent') {
      const url=new URL(p.request.url);
      if(['http:','https:','ws:','wss:'].includes(url.protocol) && url.origin!==origin) external.add(url.origin);
      if(secrets.some(s=>p.request.url.includes(s) || (p.request.postData || '').includes(s))) faults.push('Credential in URL/request body');
      requests.set(p.requestId,{path:url.pathname,method:p.request.method,body:p.request.postData});
    }
    if(event.method==='Network.responseReceived') {
      const request=requests.get(p.requestId);
      if(request) responses.push({...request,requestId:p.requestId,status:p.response.status,headers:p.response.headers,
        cached:Boolean(p.response.fromDiskCache || p.response.fromServiceWorker),done:false});
    }
    if(event.method==='Network.loadingFinished') {
      const entry=responses.find(r=>r.requestId===p.requestId && !r.done);
      if(!entry) return;
      if(entry.path.startsWith('/api/') || entry.path==='/health') {
        try {
          const payload=await cmd('Network.getResponseBody',{requestId:p.requestId});
          const raw=payload.base64Encoded ? Buffer.from(payload.body,'base64').toString() : payload.body;
          if(secrets.some(s=>raw.includes(s))) faults.push('Credential in HTTP response');
          entry.body=JSON.parse(raw);
        } catch { entry.body=null; }
      }
      entry.done=true;
    }
  });
  await cmd('Page.enable'); await cmd('Runtime.enable'); await cmd('Network.enable');
  await cmd('Page.navigate',{url:origin});
  await until(()=>evaluate(`document.readyState==='complete' && document.getElementById('mode')?.textContent.includes('TEST')`),'Shipped app TEST mode');
}

async function run() {
  await start(javaArgument());
  await test('anonymous_and_TEST_mode', async()=>{
    assert((await browserProbe('/api/me')).status===401,'Anonymous identity endpoint accessible');
    assert(await evaluate(`document.getElementById('mode').textContent.includes('TEST-ONLY') && document.body.textContent.includes('不控制设备')`),'Missing TEST/no-actuation disclosure');
    await storageCheck();
  });
  await test('wrong_token_DOM_error', async()=>{
    await fill('#token','r3-explicit-invalid-token-not-a-credential');
    const value=await submit('#login button:not(#logout)','/api/me','GET');
    assert(value.status===401 && value.body.error==='unauthorized','Wrong token was accepted');
    assert(await evaluate(`document.getElementById('message').textContent.includes('unauthorized') && document.getElementById('token').value===''`),'Wrong-token error not visible/token input not cleared');
  });
  const loggedIn=await test('real_identity_cases_tasks_and_memory_only_token',async()=>{
    await login('admin');
    const value=ok(await submit('#refresh-cases','/api/cases','GET'),'Cases');
    assert(value.some(c=>c.case_label==='normal') && value.some(c=>c.case_label==='jam') && value.every(c=>c.plant_id==='PLANT-A' && c.synthetic),'Original authorized cases missing');
    assert(await evaluate(`document.getElementById('tasks').options.length===1`),'Fresh app already has tasks');
    await storageCheck();
  });
  const published=await test('real_HermiT_checked_version_and_explicit_UI_publication',async()=>{
    const built=bootstrap.knowledge, r=built.reasoner;
    assert(built.status==='checked' && r.name==='HermiT' && r.success && r.status==='success' && r.hermit_exit_code===0 &&
      r.reload_verified && r.inferences.length>0 && /^[a-f0-9]{64}$/.test(r.java_binary_sha256) && /^[a-f0-9]{64}$/.test(r.hermit_jar_sha256),'Missing actual HermiT attestations');
    const unpublished=await browserProbe('/api/tasks','admin','POST',{
      request_id:'browser-unpublished-rejected',...Object.fromEntries(
        ['plant_id','device_id','event_id','window','complaint'].map(key=>[key,bootstrap.cases.normal[key]]))});
    assert(unpublished.status===409 && unpublished.body.error==='no_active_knowledge',
      'Checked knowledge was usable before explicit publication');
    await choose('#versions',built.version_id);
    const shown=await evaluate(`JSON.parse(document.getElementById('knowledge').textContent)`);
    assert(shown.version_id===built.version_id && shown.status==='checked','Checked version not displayed');
    await fill('#review-note','ORIGINAL SYNTHETIC only: reviewed source rules, units, real HermiT output, and exact immutable version for browser testing.');
    const value=ok(await submit('#publish button','/api/knowledge/publish'),'Explicit publication');
    assert(value.version_id===built.version_id && value.content_hash===built.content_hash,'Published a different knowledge version');
  },loggedIn);
  const normalDone=await test('normal_UI_submit_and_original_evidence',async()=>{
    normal=await newTask('normal',COMPLAINT);
    assert(normal.status==='submitted' && normal.submission?.node_id==='normal' && !normal.pending_human,'Normal did not submit without human evidence');
    assert(normal.request.complaint===COMPLAINT && normal.tree_version===bootstrap.knowledge.version_id,'Request/pinned knowledge changed');
    await taskDOM(normal);
    assert(normal.evidence.some(e=>e.source==='signal') && !normal.evidence.some(e=>e.source==='human'),'Normal signal evidence missing or invented human evidence');
  },published);
  const reportDone=await test('normal_report_exact_content_and_untrusted_TEST_draft',async()=>{
    report=ok(await submit('#make-report',`/api/tasks/${normal.task_id}/reports`),'Report generation');
    assert(report.status==='completed' && report.content_hash && report.content.authoritative.submission.content_hash===normal.submission.content_hash,'Report not bound to frozen diagnosis');
    assert(report.content.model_draft?.synthetic===true && report.content.model_draft.authority==='untrusted_model_draft' &&
      report.content.model_draft.review_status==='unreviewed' && report.signoff===null,'Synthetic narrative mistaken for authoritative/reviewed evidence');
    const shown=await evaluate(`JSON.parse(document.getElementById('report').textContent)`);
    assert(isDeepStrictEqual(shown,report),'Report DOM differs from real API');
    const task=await browserProbe(`/api/tasks/${normal.task_id}`,'admin');
    assert(isDeepStrictEqual(task.body.submission,normal.submission),'Report generation mutated frozen diagnosis');
  },normalDone);
  await test('normal_explicit_immutable_signoff',async()=>{
    await choose('#decision','approve'); await fill('#signoff-note',REVIEW);
    const value=ok(await submit('#signoff button',`/api/reports/${report.report_id}/review`),'Human signoff');
    assert(value.content_hash===report.content_hash && value.decision==='approve' && value.actor_id==='browser-admin' && value.not_actuation===true,'Signoff identity/hash mismatch');
    assert(isDeepStrictEqual(await evaluate(`JSON.parse(document.getElementById('report').textContent)`),value),'Signoff not rendered');
    const got=await browserProbe(`/api/reports/${report.report_id}`,'admin');
    assert(got.body.signoff?.signoff_id===value.signoff_id && got.body.content_hash===report.content_hash,'Signoff not persisted or content mutated');
    const repeated=await browserProbe(`/api/reports/${report.report_id}/review`,'admin','POST',
      {action_id:'browser-repeat-review',content_hash:report.content_hash,decision:'reject',note:'Explicit synthetic second review must be denied.'});
    assert(repeated.status===409 && repeated.body.error==='report_already_reviewed','Immutable review accepted a conflicting second signoff');
  },reportDone);
  await test('literal_DOM_injection_is_text',async()=>{
    const literal=await evaluate(`({submission:JSON.parse(document.getElementById('submission').textContent),
      report:JSON.parse(document.getElementById('report').textContent)})`);
    // <pre> displays JSON escaping for embedded quotes. Decode that JSON before
    // comparing original text; escaped quotes are not data corruption or HTML.
    assert(literal.submission.request.complaint===COMPLAINT &&
      (literal.report.note===REVIEW || literal.report.authoritative?.request.complaint===COMPLAINT),
      'Literal attack string did not reach actual rendered resource');
    await xssCheck();
  },reportDone);
  await test('task_switch_clears_previous_report_and_audit',async()=>{
    await openTask(normal);
    ok(await submit('#refresh-reports',`/api/tasks/${normal.task_id}/reports`,'GET'),'Normal reports');
    const mark=responses.length;
    await choose('#reports',report.report_id);
    ok(await responseAfter(mark,`/api/reports/${report.report_id}`,'GET'),'Select normal report');
    if(!await evaluate(`document.querySelector('details:has(#audit)').open`))
      await click('details:has(#audit) > summary');
    ok(await submit('#refresh-audit',`/api/tasks/${normal.task_id}/audit`,'GET'),'Normal audit before task switch');
    assert(await evaluate(`currentReport?.report_id===${lit(report.report_id)} &&
      document.getElementById('audit').textContent.includes(${lit(normal.task_id)})`),
      'Previous report/audit not actually loaded before task switch');
    jam=await newTask('jam');
    assert(await evaluate(`currentTask?.task_id===${lit(jam.task_id)} && currentReport===null && reports.length===0 &&
      ![...document.getElementById('reports').options].some(o=>o.value) &&
      !document.getElementById('report').textContent.includes(${lit(report.report_id)}) &&
      !document.getElementById('audit').textContent.includes(${lit(normal.task_id)})`),
      'Task switch retained the previous report object/choices/text or audit');
    if(!await disabled('#signoff button')) {
      const sent=[...requests.values()].filter(r=>r.path.endsWith('/review')).length;
      await fill('#signoff-note',REVIEW); await click('#signoff button'); await idle();
      assert([...requests.values()].filter(r=>r.path.endsWith('/review')).length===sent,
        'Task switch allowed signing the previous task report');
    }
  },reportDone);
  const jamPaused=await test('jam_pauses_without_inventing_human_evidence',async()=>{
    if(!jam) jam=await newTask('jam');
    assert(jam.status==='waiting_human' && !jam.submission && jam.pending_human?.condition_id==='obstruction_confirmed','Jam did not pause at the actual human boundary');
    assert(!jam.evidence.some(e=>e.source==='human'),'Human evidence invented before answer');
    const pending=jam.pending_human;
    const frozen=bootstrap.knowledge.draft.conditions.find(c=>c.condition_id===pending.condition_id);
    assert(frozen?.source==='human','Human condition missing from actual HermiT-checked knowledge');
    // Predicate is a full Condition; tree identity is carried alongside it. Compare
    // the frozen condition's fields, permitting future additional safe metadata.
    assert(pending.predicate && Object.entries(frozen).every(([key,value])=>
      isDeepStrictEqual(pending.predicate[key],value)) && pending.tree_version===jam.tree_version &&
      pending.tree_hash===jam.tree_hash, 'Human prompt predicate differs from pinned condition/tree');
    assert(typeof pending.model_question_untrusted==='string' && pending.question!==pending.model_question_untrusted,
      'Model-written question is not separated from canonical human instruction');
    const shown=await evaluate(`({hidden:document.getElementById('human').hidden,
      text:document.getElementById('human').textContent,question:document.getElementById('question').textContent,
      binding:JSON.parse(document.getElementById('human-predicate').textContent),
      observation:document.getElementById('observation').value,
      answer:document.getElementById('answer-value').value,required:document.getElementById('answer-value').required})`);
    assert(!shown.hidden && !shown.observation && shown.answer==='' && shown.required &&
      isDeepStrictEqual(shown.binding.predicate,pending.predicate) && shown.binding.tree_version===jam.tree_version &&
      shown.binding.tree_hash===jam.tree_hash, 'Pinned human predicate not visible or observation preanswered');
    assert(!pending.model_question_untrusted || !shown.question.includes(pending.model_question_untrusted),
      'Untrusted model question rendered as the authoritative human question');
    const before=await control('state');
    await delay(300);
    const unchanged=(await browserProbe(`/api/tasks/${jam.task_id}`,'admin')).body;
    assert(unchanged.status==='waiting_human' && unchanged.calls===jam.calls && (await control('state')).provider_calls===before.provider_calls,'Waiting human auto-resumed');
    await fill('#observation',OBSERVATION);
    const answerRequests=[...requests.values()].filter(r=>r.path===`/api/tasks/${jam.task_id}/answer`).length;
    await click('#answer button'); await idle();
    assert([...requests.values()].filter(r=>r.path===`/api/tasks/${jam.task_id}/answer`).length===answerRequests &&
      await evaluate(`document.getElementById('answer-value').value==='' && !document.getElementById('answer-value').checkValidity()`),
      'Unselected raw human fact submitted an answer');
    await fill('#observation','');
    const denied=await browserProbe(`/api/tasks/${jam.task_id}/reports`,'admin','POST',{action_id:'browser-premature-report'});
    assert(denied.status===409 && denied.body.error==='report_not_terminal','Report generated before terminal diagnosis');
  },published);
  const jamDone=await test('jam_explicit_synthetic_human_answer_then_submit',async()=>{
    await choose('#answer-value','true'); await fill('#observation',OBSERVATION);
    const questionId=jam.pending_human.question_id;
    const mark=responses.length;
    ok(await submit('#answer button',`/api/tasks/${jam.task_id}/answer`),'Explicit synthetic human answer');
    const answered=responses.slice(mark).find(r=>r.path===`/api/tasks/${jam.task_id}/answer` && r.method==='POST');
    const sent=JSON.parse(requests.get(answered?.requestId)?.body || 'null');
    assert(sent?.question_id===questionId && sent.answer===true && sent.observation_note===OBSERVATION,
      'UI did not submit the explicit raw boolean against the current question ID');
    jam=ok(await responseAfter(mark,`/api/tasks/${jam.task_id}/run`),'Post-answer run');
    assert(jam.status==='submitted' && jam.submission?.node_id==='jam' && !jam.pending_human,'Jam did not submit after explicit answer');
    const human=jam.evidence.find(e=>e.condition_id==='obstruction_confirmed');
    assert(human?.source==='human' && human.state==='supported' && human.details.answer===true &&
      human.details.question_id===questionId && human.details.observation_note===OBSERVATION,
      'Raw synthetic observation/provenance changed or model question inverted the answer');
    await taskDOM(jam); await xssCheck();
  },jamPaused);
  await test('real_audit_and_no_fixture_oracle_in_requests',async()=>{
    if(!await evaluate(`document.querySelector('details:has(#audit)').open`))
      await click('details:has(#audit) > summary');
    const audit=ok(await submit('#refresh-audit',`/api/tasks/${jam.task_id}/audit`,'GET'),'Audit');
    assert(audit.events.length>0 && audit.model_calls.length>0 && isDeepStrictEqual(await evaluate(`JSON.parse(document.getElementById('audit').textContent)`),audit),'Actual audit not rendered');
    for(const request of requests.values()) if(request.path==='/api/tasks' && request.method==='POST') {
      const value=JSON.parse(request.body);
      assert(!('case_label' in value) && !('expected' in value) && !('diagnosis' in value),'Fixture oracle leaked to diagnosis request');
    }
    await storageCheck();
  },jamDone);
  const unknownCreated=await test('unknown_result_no_automatic_retry_and_guarded_controls',async()=>{
    const initial=await control('state');
    await control('unknown_once'); unknown=await newTask('normal');
    assert(unknown.status==='blocked' && !!unknown.pending_call_id && unknown.error.code==='provider_unknown',
      'Explicit synthetic timeout did not use real unknown service boundary');
    const before=await control('state');
    assert(before.provider_calls===initial.provider_calls+1 && before.synthetic_decisions===initial.synthetic_decisions &&
      before.injected_timeouts===initial.injected_timeouts+1 && unknown.calls===1 && unknown.tokens===0,
      'Zero-usage fixture did not stop before the deterministic provider decision');
    await delay(300);
    assert((await control('state')).provider_calls===before.provider_calls,'Unknown call automatically retried');
    const guarded=await browserProbe(`/api/tasks/${unknown.task_id}/run`,'admin','POST',{});
    assert(guarded.body.status==='blocked' && guarded.body.pending_call_id===unknown.pending_call_id &&
      (await control('state')).provider_calls===before.provider_calls,'Explicit run resent an unresolved call');
    assert(await disabled('#resume') && await evaluate(`!document.getElementById('reconcile').closest('[hidden]')`),
      'Unknown-outcome resume/reconciliation controls not safely gated');
    assert(await evaluate(`document.getElementById('call-id').value===${lit(unknown.pending_call_id)} && !document.getElementById('no-result').checked`),'Unknown call ID missing or no-result human attestation prechecked');
    assert(await evaluate(`(() => {const e=document.getElementById('reconcile-tokens');
      return e && e.type==='number' && e.min==='0' && e.max==='100000' && e.step==='1' && !e.required && e.value==='';})()`),
      'Optional usage input lacks integer bounds or silently defaults unknown usage');
  },published);
  await test('unknown_explicit_owner_admin_reconciliation',async()=>{
    const note='ORIGINAL SYNTHETIC zero-usage attestation: checked fixture counters show one injected timeout before SyntheticProvider.decision, no deterministic decision or model inference ran; known total token usage for this call is 0. No usable result exists. This is a synthetic test attestation, not external execution proof.';
    await fill('#reconcile-note',note); await fill('#reconcile-tokens','0');
    await click('#no-result');
    const response=await submit('#reconcile button',`/api/tasks/${unknown.task_id}/reconcile`);
    const value=ok(response,'Explicit zero-usage reconciliation');
    const sent=JSON.parse(requests.get(response.requestId)?.body || 'null');
    assert(sent?.call_id===unknown.pending_call_id && sent.no_result===true && sent.tokens===0 && sent.note===note,
      'UI did not transmit the explicit known-zero usage and exact synthetic attestation');
    assert(!value.pending_call_id && value.calls===unknown.calls && value.tokens===unknown.tokens,
      'Known-zero reconciliation reset consumed budgets or retained pending call');
    assert(await evaluate(`document.getElementById('reconcile-tokens').value===''`),'Usage input was not cleared after reconciliation');
    await reconciliationAudit(unknown,0,false,note);
    const resumed=ok(await submit('#resume',`/api/tasks/${unknown.task_id}/run`),'Explicit post-reconciliation run');
    assert(resumed.status==='submitted' && resumed.submission.node_id==='normal' &&
      resumed.calls>unknown.calls && resumed.tokens>unknown.tokens && resumed.tokens<resumed.limits.tokens,
      'Explicit known-zero reconciliation did not permit bounded, accounted continuation');
    assert(await disabled('#reconcile button'),'Reconcile control enabled without an unknown call');
  },unknownCreated);
  await test('unknown_blank_usage_conservatively_exhausts_budget',async()=>{
    await click('#logout'); await login('admin');
    const initial=await control('state');
    await control('unknown_once'); unattested=await newTask('normal');
    const before=await control('state');
    assert(unattested.status==='blocked' && !!unattested.pending_call_id && unattested.error.code==='provider_unknown' &&
      before.provider_calls===initial.provider_calls+1 && before.synthetic_decisions===initial.synthetic_decisions &&
      before.injected_timeouts===initial.injected_timeouts+1 && unattested.tokens<unattested.limits.tokens,
      'Blank-usage case did not reach the real unresolved call boundary with remaining budget');
    assert(await evaluate(`document.getElementById('reconcile-tokens').value==='' && !document.getElementById('no-result').checked`),
      'Fresh unknown-call form prefilled usage or confirmed no result');
    const note='ORIGINAL SYNTHETIC omitted-usage accounting test: the injected timeout has no usable result. Deliberately supply no verified usage attestation, even though the test fixture is controlled; the application must conservatively charge remaining budget rather than infer zero.';
    await fill('#reconcile-note',note); await click('#no-result');
    const response=await submit('#reconcile button',`/api/tasks/${unattested.task_id}/reconcile`);
    const value=ok(response,'Explicit reconciliation without usage');
    const sent=JSON.parse(requests.get(response.requestId)?.body || 'null');
    assert(sent?.call_id===unattested.pending_call_id && sent.no_result===true && !Object.hasOwn(sent,'tokens') && sent.note===note,
      'Blank usage was not omitted from the real UI request');
    const charged=unattested.limits.tokens-unattested.tokens;
    assert(!value.pending_call_id && value.status==='ready' && value.calls===unattested.calls &&
      value.tools===unattested.tools && value.tokens===unattested.tokens+charged,
      'Omitted usage failed to conservatively consume the remaining token budget');
    // Replay the exact captured UI command via real browser HTTP, without a second
    // human declaration. The durable receipt must not charge that action twice.
    const replay=ok(await browserProbe(`/api/tasks/${unattested.task_id}/reconcile`,'admin','POST',sent),'Same-action reconciliation replay');
    assert(isDeepStrictEqual(replay,value),'Same-action reconciliation replay changed the accounted state');
    await reconciliationAudit(unattested,charged,true,note);
    const resumed=ok(await submit('#resume',`/api/tasks/${unattested.task_id}/run`),'Explicit budget-exhausted continuation');
    assert(resumed.status==='escalated' && resumed.error?.code==='budget_exhausted' && !resumed.submission &&
      !resumed.pending_call_id && resumed.tokens===value.tokens && resumed.calls===value.calls &&
      resumed.tools===value.tools && (await control('state')).provider_calls===before.provider_calls,
      'Unattested usage allowed another provider attempt or failed to escalate at exhausted budget');
    assert(await disabled('#resume') && await disabled('#reconcile button') && await evaluate(
      `document.getElementById('task-status').textContent.includes('escalated') && document.getElementById('budget').textContent.includes(${lit(`token ${resumed.tokens}/${resumed.limits.tokens}`)})`),
      'Budget-exhausted terminal state or guarded controls not reflected in the UI');
  },published);
  await test('operator_role_controls_and_owner_isolation',async()=>{
    await click('#logout'); await login('operator');
    const mine=ok(await submit('#refresh-tasks','/api/tasks','GET'),'Operator tasks');
    assert(mine.length===0,'Operator list exposed another owner');
    if(normal) for(const path of [`/api/tasks/${normal.task_id}`,`/api/tasks/${normal.task_id}/audit`])
      assert((await browserProbe(path,'operator')).status===404,'Operator read another owner task');
    if(report) assert((await browserProbe(`/api/reports/${report.report_id}`,'operator')).status===404,'Operator read another owner report');
    assert(await disabled('#publish button') && await disabled('#signoff button') && await disabled('#reconcile button'),'Operator has enabled admin/author-only controls');
  },loggedIn);
  await test('invalid_replacement_login_clears_old_identity',async()=>{
    await click('#logout'); await login('admin');
    if(normal) await openTask(normal);
    await fill('#token','r3-replacement-invalid-token-not-a-credential');
    const response=await submit('#login button:not(#logout)','/api/me','GET');
    assert(response.status===401,'Replacement invalid token accepted');
    await cleared(); await storageCheck();
  },loggedIn);
  await test('logout_clears_token_and_sensitive_DOM',async()=>{
    await click('#logout'); await login('admin'); if(jamDone) await openTask(jam);
    for(const id of ['complaint','review-note','signoff-note','reconcile-note'])
      await fill('#'+id,`ORIGINAL SYNTHETIC sensitive input ${id}`);
    await fill('#reconcile-tokens','37');
    await click('#logout'); await cleared(); await storageCheck();
  },loggedIn);
  await test('logout_rejects_late_real_response',async()=>{
    await login('admin'); await control('hold_cases');
    const mark=responses.length;
    try {
      await click('#refresh-cases'); await until(async()=>(await control('state')).held,'Actual API response held');
      await click('#logout');
      const message=await evaluate(`document.getElementById('message').textContent`);
      await control('release_cases');
      const late=await responseAfter(mark,'/api/cases','GET');
      assert(late.status===200 && late.body.length>0,'Race did not deliver a real old authorized response');
      await idle(); await cleared();
      assert(await evaluate(`document.getElementById('message').textContent===${lit(message)}`),'Late response/error overwrote post-logout message');
      const sent=requests.size;
      // Authentication-free state must not reuse the old in-memory header.
      assert(await evaluate(`token===''`),'Late response restored credential');
      await storageCheck(); assert(requests.size===sent,'Storage probe unexpectedly made an app request');
    } finally { await control('release_cases'); }
  },loggedIn);
  await test('reload_forgets_identity',async()=>{
    await login('admin');
    await cmd('Page.reload',{ignoreCache:false});
    await until(()=>evaluate(`document.readyState==='complete' && document.getElementById('mode')?.textContent.includes('TEST')`),'Reloaded app');
    await cleared(); await storageCheck();
    assert((await browserProbe('/api/me')).status===401,'Reload retained usable authentication');
  },loggedIn);
  await test('security_headers_on_success_and_boundary_errors',async()=>{
    for(const path of ['/health','/static/app.js','/static/style.css','/','/api/me']) {
      const value=await evaluate(`(async()=>{const r=await fetch(${lit(path)},{cache:'no-store',credentials:'omit'}); await r.text(); return Object.fromEntries(r.headers);})()`);
      headersCheck(value,path);
    }
    const wrongType=await browserProbe('/api/tasks','admin','POST',{}, {'Content-Type':'text/plain'});
    assert(wrongType.status===415,'Non-JSON body accepted'); headersCheck(wrongType.headers,'415 json_required');
    const oversized=await browserProbe('/api/tasks','admin','POST',{padding:'x'.repeat(256001)});
    assert(oversized.status===413,'Oversized body accepted'); headersCheck(oversized.headers,'413 body_too_large');
    const invalid=await browserProbe('/api/tasks','admin','POST',{unexpected:true});
    assert(invalid.status===422,'Malformed request accepted'); headersCheck(invalid.headers,'422 invalid_request');
    // Browser fetch forbids Host/Origin and Node fetch may rewrite Host. Send only
    // these two credential-free negatives via built-in HTTP to the loopback socket.
    for(const [headers,status,label] of [[{Host:'example.invalid'},400,'invalid_host'],[{Origin:'https://example.invalid'},403,'cross_origin_denied']]) {
      const response=await new Promise((resolve,reject)=>{
        const request=httpRequest(origin+'/health',{headers,agent:false},response=>{
          response.resume(); response.once('end',()=>resolve({status:response.statusCode,headers:response.headers}));
          response.once('error',reject);
        });
        request.setTimeout(5000,()=>request.destroy(new Error('Loopback header probe timeout')));
        request.once('error',reject); request.end();
      });
      assert(response.status===status,`${label}: expected ${status}, received ${response.status}`);
      headersCheck(response.headers,label);
    }
  });
  await test('no_external_requests_console_tokens_or_persistent_credentials',async()=>{
    await storageCheck(); await xssCheck();
    const api=responses.filter(r=>r.path.startsWith('/api/'));
    assert(api.length>10 && api.every(r=>!r.cached),'Insufficient real API responses or cache reuse');
    for(const response of api) headersCheck(response.headers,`${response.method} ${response.path}`);
    assert(!faults.length,faults.join('; ')); assert(!external.size,'Page attempted non-app network traffic');
  });
}

function killGroup(child, signal) {
  if(!child?.pid || child.launchFailed || child.exitCode!==null || child.signalCode!==null) return;
  try { process.kill(-child.pid,signal); } catch(error) { if(error.code!=='ESRCH') throw error; }
}
async function processTable() {
  const {stdout}=await promisify(execFile)('/bin/ps',['-axo','pid=,ppid=,lstart='],
    {maxBuffer:2*1024*1024,timeout:3000});
  return stdout.trim().split('\n').map(line=>{
    const [pid,parent,...started]=line.trim().split(/\s+/);
    return {pid:Number(pid),parent:Number(parent),started:started.join(' ')};
  });
}
async function ownedDescendants() {
  // HermiT deliberately starts a worker process group. Identify ONLY descendants of
  // our live direct children before terminating; never use pkill/killall/name matching.
  const rows=await processTable();
  const pids=new Set(children.filter(p=>p.pid && p.exitCode===null && p.signalCode===null).map(p=>p.pid));
  let changed=true;
  while(changed) { changed=false; for(const row of rows) if(pids.has(row.parent) && !pids.has(row.pid)) {pids.add(row.pid);changed=true;} }
  return rows.filter(row=>pids.has(row.pid) && !children.some(p=>p.pid===row.pid)).reverse();
}
async function cleanup() {
  stopping=true; cdp?.close();
  for(const pending of controls.values()) { clearTimeout(pending.timer); pending.reject(new Error('Fixture cleanup')); }
  controls.clear();
  let descendants=[], snapshotError;
  try { if(children.length) descendants=await ownedDescendants(); }
  catch(error) { snapshotError=error; }
  try {
    // Even if process enumeration fails, always terminate our direct child groups.
    for(const {pid} of descendants) { try {process.kill(pid,'SIGTERM');} catch(e) {if(e.code!=='ESRCH') throw e;} }
    for(const child of children) killGroup(child,'SIGTERM');
    await Promise.race([Promise.all(children.map(child=>child.exitCode!==null || child.signalCode!==null || child.launchFailed
      ? Promise.resolve() : new Promise(resolve=>child.once('exit',resolve)))),delay(2000)]);
    for(const child of children) killGroup(child,'SIGKILL');
    if(descendants.length) {
      const current=await processTable();
      // Verify start time too, so a recycled PID cannot become a cleanup target.
      for(const owned of descendants) if(current.some(row=>row.pid===owned.pid && row.started===owned.started)) {
        try {process.kill(owned.pid,'SIGKILL');} catch(e) {if(e.code!=='ESRCH') throw e;}
      }
    }
    await delay(200);
    assert(children.every(p=>p.exitCode!==null || p.signalCode!==null || p.launchFailed),'Owned child did not exit');
    if(descendants.length) {
      const remaining=await processTable();
      assert(!descendants.some(owned=>remaining.some(row=>row.pid===owned.pid && row.started===owned.started)),
        'Owned descendant process did not exit');
    }
    if(snapshotError) throw new Error('Owned descendant enumeration failed; direct child groups were stopped');
    record('owned_process_cleanup','pass');
  } catch(error) {
    for(const child of children) { try {killGroup(child,'SIGKILL');} catch {} }
    record('owned_process_cleanup','fail',error.message);
  }
  if(root) {
    try {
      let count=0;
      const needles=secrets.flatMap(s=>[Buffer.from(s),Buffer.from(s,'utf16le')]);
      async function scan(path) {
        for(const entry of await readdir(path,{withFileTypes:true})) {
          if(entry.isSymbolicLink()) continue; // Chrome lock links are never followed.
          const file=join(path,entry.name);
          if(entry.isDirectory()) await scan(file);
          else if(entry.isFile()) { const bytes=await readFile(file); count++;
            assert(!needles.some(n=>bytes.includes(n)),'Plaintext credential found in generated app/profile file (content withheld)'); }
        }
      }
      await scan(root);
      record('no_plaintext_tokens_in_owned_files','pass',`${count} generated files scanned as UTF-8/UTF-16 bytes after shutdown; no content logged`);
    } catch(error) { record('no_plaintext_tokens_in_owned_files','fail',error.message); }
    try { await rm(root,{recursive:true,force:true,maxRetries:3,retryDelay:100}); record('owned_temp_app_profile_cleanup','pass'); }
    catch { record('owned_temp_app_profile_cleanup','fail','Could not delete owned temporary directory'); }
  }
}
function output() {
  for(const name of UI_CASES) if(!results.some(r=>r.name===name)) record(name,'not_tested','Browser/bootstrap prerequisite failed');
  record('scope_not_tested','not_tested','No model inference, managed provider, production/physical diagnosis, fresh package install, other browsers, wheel build, upload/extraction UI, knowledge rejection, all remaining telemetry cases, ACL revocation, hard-crash durability/expired leases, active-worker 409, multi-tab races, previously recorded positive-usage reduction/delta accounting, or adversarial inverted human questions/false-threshold predicates. Jam checks raw true against the canonical pinned predicate, not free question wording. Unknown calls use explicit synthetic pre-decision timeouts; explicit zero and omitted usage are separate browser cases, with an exact captured-action HTTP replay for receipt idempotence. Logout race delays actual ASGI output. No screenshots retained.');
  const counts=Object.fromEntries(['pass','fail','not_tested'].map(s=>[s,results.filter(r=>r.status===s).length]));
  process.stdout.write(JSON.stringify({status:counts.fail?'fail':'pass',mode:'Actual installed Chrome CDP/WebSocket + real local API; SyntheticProvider TEST ONLY, original telemetry, actual HermiT',
    elapsed_seconds:Number(((Date.now()-START)/1000).toFixed(2)),counts,results},null,2)+'\n');
  return counts.fail ? 1 : 0;
}
let ending=false;
async function finish(reason) {
  if(ending) return;
  ending=true;
  if(reason) record('browser_acceptance','fail',reason);
  await cleanup(); clearTimeout(hardTimer);
  process.exitCode=output();
}
const hardTimer=setTimeout(()=>{void finish('285s absolute acceptance deadline');},285000);
for(const signal of ['SIGINT','SIGTERM','SIGHUP']) process.once(signal,()=>{void finish(`Interrupted by ${signal}`);});
try { await run(); } catch(error) { record('browser_acceptance','fail',error.message); }
finally { await finish(); }
