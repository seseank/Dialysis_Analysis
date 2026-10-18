from flask import Flask, jsonify, render_template_string
import genai as g

app = Flask(__name__)
ranked = g.today.sort_values("risk", ascending=False).head(15)
by_id = {int(r.patient_id): r for r in ranked.itertuples()}
rows = [dict(id=int(r.patient_id), name=r.name, risk=round(r.risk * 100), clinic=r.clinic,
             language=r.language, transport=r.transport_type.replace("_", " "),
             distance=r.distance_mi, why=", ".join(g.reasons(r))) for r in ranked.itertuples()]
w = g.today.iloc[0]

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Dialysis Attendance Risk</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f9;color:#1d2433}
header{background:#0f62fe;color:#fff;padding:20px 28px}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.85}
.stats{display:flex;gap:16px;padding:20px 28px;flex-wrap:wrap}
.card{background:#fff;border-radius:10px;padding:14px 18px;box-shadow:0 1px 3px #0001;min-width:150px;font-size:13px;color:#5a6275}
.card b{display:block;font-size:24px;color:#1d2433}
table{width:calc(100% - 56px);margin:0 28px 28px;border-collapse:collapse;background:#fff;box-shadow:0 1px 3px #0001}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #eef0f4;font-size:14px;vertical-align:top}
th{background:#f8f9fb;font-size:12px;text-transform:uppercase;color:#5a6275}
.risk{font-weight:700;padding:3px 8px;border-radius:12px;color:#fff}.high{background:#da1e28}.med{background:#f1c21b;color:#1d2433}
button{background:#0f62fe;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer}
.msg{margin-top:6px;font-style:italic}
</style></head><body>
<header><h1>Dialysis Attendance Risk: {{date}}</h1><p>Predictive model + IBM Granite on IBM LinuxONE</p></header>
<div class=stats>
<div class=card>Patients scheduled<b>{{total}}</b></div>
<div class=card>High risk (50%+)<b>{{high}}</b></div>
<div class=card>Rain<b>{{rain}} in</b></div>
<div class=card>High / Low<b>{{tmax}}° / {{tmin}}°F</b></div></div>
<table><tr><th>Patient</th><th>Risk</th><th>Why flagged</th><th>Transport</th><th>Clinic</th><th>Outreach</th></tr>
{% for p in rows %}<tr><td>{{p.name}}<br><small>{{p.language}}</small></td>
<td><span class="risk {{'high' if p.risk>=50 else 'med'}}">{{p.risk}}%</span></td>
<td>{{p.why}}</td><td>{{p.transport}}<br><small>{{p.distance}} mi</small></td><td><small>{{p.clinic}}</small></td>
<td><button onclick="gen({{p.id}},this)">Generate message</button><div class=msg id=m{{p.id}}></div></td></tr>{% endfor %}
</table><script>
async function gen(id,b){b.disabled=true;b.textContent='Granite is writing...';
const d=await (await fetch('/message/'+id)).json();
document.getElementById('m'+id).textContent=d.message;b.textContent='Regenerate';b.disabled=false;}
</script></body></html>"""

@app.route("/")
def home():
    return render_template_string(PAGE, rows=rows, date=w.date.strftime("%b %d, %Y"),
        total=len(g.today), high=int((g.today.risk >= 0.5).sum()),
        rain=round(w.rain_in, 2), tmax=int(w.temp_max), tmin=int(w.temp_min))

@app.route("/message/<int:pid>")
def message(pid):
    r = by_id[pid]
    why = g.reasons(r)
    return jsonify(message=g.llm(g.build_prompt(r, why)) or g.fallback(r, why))

app.run(host="0.0.0.0", port=5000, threaded=False)
