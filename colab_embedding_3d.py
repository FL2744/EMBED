"""
HOW TO RUN THIS VISUALIZATION
=============================

First-time local setup (run in the VS Code terminal):
    cd embedding-visualizer
    python3 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    ./.venv/bin/python -m pip install numpy ipython ipykernel

VS Code interpreter:
    1. Press Cmd+Shift+P.
    2. Choose "Python: Select Interpreter".
    3. Select:
       <repository-path>/.venv/bin/python

Run locally from the terminal:
    cd embedding-visualizer
    ./.venv/bin/python colab_embedding_3d.py

You can also use the VS Code Run/Debug button after selecting the interpreter.
When run locally, the script writes the interactive visualization to:
    <repository-path>/embedding_visualization.html

The HTML visualization should open automatically in the default browser. If it
does not, open embedding_visualization.html manually.

Run in Google Colab:
    1. Upload colab_embedding_3d.py to the Colab Files panel.
    2. Run this in a notebook cell:
       %run /content/colab_embedding_3d.py

In Colab/Jupyter/VS Code notebooks the visualization renders inside the notebook.
In a normal Python terminal or VS Code debugger it opens as a browser page.
"""

import html as html_module
import json
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from IPython.display import HTML, display


def example_embedding(seed=42, spread_scale=5.0):
    """Return one deterministic example 768-dimensional embedding."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(768) * spread_scale


def embedding_to_3d_path(vector, turn_angle_degrees=45.0):
    """Create a 3-D path with sign-controlled right/left fixed-angle turns."""
    vector = np.asarray(vector, dtype=float)
    alpha = np.deg2rad(turn_angle_degrees)
    points = [np.zeros(3)]
    direction = np.array([1.0, 0.0, 0.0])

    world_up = np.array([0.0, 0.0, 1.0])
    for i, value in enumerate(vector):
        # For a heading along +x, cross(heading, up) points toward -y: right.
        local_right = np.cross(direction, world_up)
        if np.linalg.norm(local_right) < 1e-10:
            local_right = np.cross(direction, [0.0, 1.0, 0.0])
        local_right /= np.linalg.norm(local_right)
        local_up = np.cross(local_right, direction)
        local_up /= np.linalg.norm(local_up)

        # Slowly roll the local turning plane to make the path genuinely 3-D.
        # Roll is independent of the embedding sign.
        roll = 0.72 * np.sin(i * np.pi * (3.0 - np.sqrt(5.0)))
        turn_side = np.cos(roll) * local_right + np.sin(roll) * local_up
        turn_side *= 1.0 if value >= 0 else -1.0

        new_direction = (
            np.cos(alpha) * direction + np.sin(alpha) * turn_side
        )
        new_direction /= np.linalg.norm(new_direction)
        points.append(points[-1] + abs(value) * new_direction)
        direction = new_direction

    return np.asarray(points)


def normalize_points(points, radius=42.0):
    points = np.asarray(points, dtype=float)
    centered = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = float(np.max(np.ptp(centered, axis=0)))
    return centered if span == 0 else centered * (2.0 * radius / span)


def show_embedding_3d(embedding=None):
    """Display in Google Colab without external JavaScript dependencies."""
    if embedding is None:
        embedding = example_embedding()

    points = normalize_points(embedding_to_3d_path(embedding))
    element_id = "embedding_" + uuid.uuid4().hex

    body_html = f"""
    <div id="{element_id}" style="position:relative;width:100%;height:720px;
         background:#060914;border-radius:14px;overflow:hidden;">
      <canvas style="display:block;width:100%;height:100%;touch-action:none"></canvas>
      <div style="position:absolute;left:14px;top:12px;color:#e8edff;
           font:13px system-ui;pointer-events:none">
        <b>768-dimensional embedding</b><br>
        Length = |value| · positive turns right · negative turns left<br>
        Consecutive angle = <span data-readout="angle">45</span>° · index-driven 3D roll<br>
        Click a point or line to deform · drag to orbit · wheel to zoom
        <div data-readout="status" style="margin-top:4px;color:#8ee6a8">Loading canvas…</div>
      </div>
      <div class="embedding-controls" style="position:absolute;right:12px;top:12px;width:210px;
        padding:10px;background:rgba(9,14,32,.88);color:#e8edff;border:1px solid #26345c;
        border-radius:10px;font:12px system-ui;box-shadow:0 6px 22px #0008">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
          <button data-action="pause">Pause</button><button data-action="reset">Reset</button>
          <button data-action="twist-left">Twist −</button><button data-action="twist-right">Twist +</button>
          <button data-action="contract">Contract</button><button data-action="expand">Expand</button>
          <button data-action="smooth">Smooth</button><button data-action="jitter">Deform</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 68px;gap:6px;align-items:end;
          padding-bottom:5px;border-bottom:1px solid #26345c;margin-bottom:6px">
          <label style="display:block;margin:0">Path points
            <input data-control="count" type="number" min="2" max="769" step="1" value="769"
              style="box-sizing:border-box;width:100%;margin-top:2px;background:#0d1730;color:#fff;
              border:1px solid #40527e;border-radius:5px;padding:4px">
          </label>
          <button data-action="apply-count">Apply</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 68px;gap:6px;align-items:end;
          padding-bottom:5px;border-bottom:1px solid #26345c;margin-bottom:6px">
          <label style="display:block;margin:0">Consecutive angle (°)
            <input data-control="angle" type="number" min="1" max="179" step="1" value="45"
              style="box-sizing:border-box;width:100%;margin-top:2px;background:#0d1730;color:#fff;
              border:1px solid #40527e;border-radius:5px;padding:4px">
          </label>
          <button data-action="apply-angle">Apply</button>
        </div>
        <label style="display:block;margin:5px 0">Rotation speed <input data-control="speed" style="width:100%" type="range" min="-3" max="3" step=".05" value="1"></label>
        <label style="display:block;margin:5px 0">Deform strength <input data-control="strength" style="width:100%" type="range" min=".5" max="12" step=".5" value="4.5"></label>
        <label style="display:block;margin:5px 0">Deform radius <input data-control="radius" style="width:100%" type="range" min="2" max="40" step="1" value="12"></label>
        <label style="display:block;margin:5px 0">Line width <input data-control="line" style="width:100%" type="range" min=".25" max="5" step=".25" value="1.25"></label>
        <label style="display:block;margin:5px 0">Point size <input data-control="point" style="width:100%" type="range" min="0" max="6" step=".25" value="1.15"></label>
        <label style="display:block;margin:5px 0">Zoom <input data-control="zoom" style="width:100%" type="range" min="1.5" max="15" step=".1" value="5.8"></label>
        <div style="display:flex;justify-content:space-between;margin-top:7px">
          <label>Line <input data-control="line-color" type="color" value="#65d9ff"></label>
          <label>Points <input data-control="point-color" type="color" value="#ffd166"></label>
        </div>
      </div>
    </div>
    """

    # Colab executes Javascript output, whereas scripts inside HTML output are sanitized.
    js = r"""
    (() => {
      try {
      const host = document.getElementById(ELEMENT_ID);
      if (!host) throw new Error('Embedding container was not created.');
      const canvas = host.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const getControl = name => host.querySelector(`[data-control="${name}"]`);
      const getAction = name => host.querySelector(`[data-action="${name}"]`);
      const status = host.querySelector('[data-readout="status"]');
      const embedding = EMBEDDING_DATA;
      let embeddingBasePath = POINT_DATA;
      let pts = embeddingBasePath.map(p => p.slice());
      let yaw = -0.55, pitch = 0.52, spin = 0, zoom = 5.8;
      let rotationSpeed=1, deformStrength=4.5, deformRadius=12;
      let lineWidth=1.25, pointSize=1.15, lineColor='#65d9ff', pointColor='#ffd166';
      let paused=false;
      let dragging = false, moved = false, lastX = 0, lastY = 0;
      let projected = [];

      function embeddingPath(values, angleDegrees) {
        const alpha=angleDegrees*Math.PI/180, ca=Math.cos(alpha), sa=Math.sin(alpha);
        const result=[[0,0,0]]; let direction=[1,0,0];
        const norm=a=>Math.hypot(a[0],a[1],a[2]);
        const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
        for (let i=0;i<values.length;i++) {
          const sign=values[i]>=0 ? 1 : -1;
          let localRight=cross(direction,[0,0,1]);
          if (norm(localRight)<1e-10) localRight=cross(direction,[0,1,0]);
          let n=norm(localRight); localRight=localRight.map(v=>v/n);
          let localUp=cross(localRight,direction);
          n=norm(localUp); localUp=localUp.map(v=>v/n);
          const roll=.72*Math.sin(i*Math.PI*(3-Math.sqrt(5)));
          const turnSide=localRight.map((v,k)=>sign*(Math.cos(roll)*v+Math.sin(roll)*localUp[k]));
          let next=direction.map((v,k)=>ca*v+sa*turnSide[k]);
          n=norm(next); next=next.map(v=>v/n);
          const previous=result[result.length-1], length=Math.abs(values[i]);
          result.push(previous.map((v,k)=>v+length*next[k])); direction=next;
        }
        return result;
      }

      function normalizePath(source, radius=42) {
        const mins=[Infinity,Infinity,Infinity], maxs=[-Infinity,-Infinity,-Infinity];
        for (const p of source) for (let k=0;k<3;k++) {
          mins[k]=Math.min(mins[k],p[k]); maxs[k]=Math.max(maxs[k],p[k]);
        }
        const center=mins.map((v,k)=>(v+maxs[k])/2);
        const span=Math.max(...mins.map((v,k)=>maxs[k]-v)) || 1;
        return source.map(p=>p.map((v,k)=>(v-center[k])*(2*radius/span)));
      }

      function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const rect = host.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function rotatePoint(p) {
        // Animated z rotation, followed by user-controlled x/y orbit.
        let c = Math.cos(spin), s = Math.sin(spin);
        let x = c*p[0] - s*p[1], y = s*p[0] + c*p[1], z = p[2];
        c = Math.cos(yaw); s = Math.sin(yaw);
        let x2 = c*x + s*z, z2 = -s*x + c*z;
        c = Math.cos(pitch); s = Math.sin(pitch);
        return [x2, c*y - s*z2, s*y + c*z2];
      }

      function project(p, w, h) {
        const q = rotatePoint(p);
        const cameraDistance = 150;
        const perspective = cameraDistance / Math.max(35, cameraDistance - q[2]);
        return [w/2 + q[0]*zoom*perspective,
                h/2 - q[1]*zoom*perspective, q[2], perspective];
      }

      function draw() {
        const w = host.clientWidth, h = host.clientHeight;
        ctx.clearRect(0, 0, w, h);
        projected = pts.map(p => project(p, w, h));

        const glow = ctx.createLinearGradient(0, 0, w, h);
        glow.addColorStop(0, '#7de8ff'); glow.addColorStop(1, '#716bff');
        ctx.lineWidth = 1.25; ctx.strokeStyle = glow;
        ctx.shadowColor = '#34cfff'; ctx.shadowBlur = 4;
        ctx.lineWidth = lineWidth; ctx.strokeStyle = lineColor;
        ctx.shadowColor = lineColor; ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.moveTo(projected[0][0], projected[0][1]);
        for (let i=1; i<projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
        ctx.stroke();

        ctx.shadowBlur = 3; ctx.fillStyle = '#ffd166';
        ctx.shadowBlur = 3; ctx.fillStyle = pointColor;
        for (const p of projected) {
          if (pointSize <= 0) break;
          const r = Math.max(.4, Math.min(8, pointSize*p[3]));
          ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2*Math.PI); ctx.fill();
        }
      }

      function pointSegmentDistance(px, py, a, b) {
        const vx=b[0]-a[0], vy=b[1]-a[1], wx=px-a[0], wy=py-a[1];
        const vv=vx*vx+vy*vy;
        const t=vv ? Math.max(0, Math.min(1, (wx*vx+wy*vy)/vv)) : 0;
        return Math.hypot(px-(a[0]+t*vx), py-(a[1]+t*vy));
      }

      function deform(index) {
        const radius=deformRadius, strength=deformStrength;
        const p=pts[Math.max(0, Math.min(index, pts.length-1))];
        let push=[-p[1], p[0], 24];
        const n=Math.hypot(...push) || 1; push=push.map(v => v/n);
        for (let j=Math.max(1,index-radius); j<=Math.min(pts.length-1,index+radius); j++) {
          const w=0.5+0.5*Math.cos(Math.PI*(j-index)/radius);
          for (let k=0;k<3;k++) pts[j][k] += push[k]*strength*w;
        }
      }

      function transformShape(kind) {
        const center=[0,0,0];
        for (const p of pts) for (let k=0;k<3;k++) center[k]+=p[k]/pts.length;
        if (kind==='expand' || kind==='contract') {
          const scale=kind==='expand' ? 1.12 : 0.89;
          for (const p of pts) for (let k=0;k<3;k++) p[k]=center[k]+(p[k]-center[k])*scale;
        } else if (kind==='twist-left' || kind==='twist-right') {
          const sign=kind==='twist-right' ? 1 : -1;
          for (let i=0;i<pts.length;i++) {
            const a=sign*(i/(pts.length-1)-.5)*.65, c=Math.cos(a), s=Math.sin(a);
            const x=pts[i][0]-center[0], y=pts[i][1]-center[1];
            pts[i][0]=center[0]+c*x-s*y; pts[i][1]=center[1]+s*x+c*y;
          }
        } else if (kind==='smooth') {
          const next=pts.map(p=>p.slice());
          for (let i=1;i<pts.length-1;i++) for (let k=0;k<3;k++)
            next[i][k]=.25*pts[i-1][k]+.5*pts[i][k]+.25*pts[i+1][k];
          pts=next;
        } else if (kind==='jitter') {
          const index=1+Math.floor(Math.random()*(pts.length-2)); deform(index);
        }
      }

      function resamplePath(source, count) {
        // Evenly resample by cumulative arc length, preserving the overall path.
        count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(count)));
        const cumulative=[0];
        for (let i=1;i<source.length;i++)
          cumulative.push(cumulative[i-1]+Math.hypot(
            source[i][0]-source[i-1][0], source[i][1]-source[i-1][1], source[i][2]-source[i-1][2]));
        const total=cumulative[cumulative.length-1];
        if (!total) return Array.from({length:count},()=>source[0].slice());
        const result=[]; let segment=1;
        for (let n=0;n<count;n++) {
          const target=total*n/(count-1);
          while (segment<cumulative.length-1 && cumulative[segment]<target) segment++;
          const a=source[segment-1], b=source[segment];
          const span=cumulative[segment]-cumulative[segment-1];
          const t=span ? (target-cumulative[segment-1])/span : 0;
          result.push(a.map((v,k)=>v+(b[k]-v)*t));
        }
        return result;
      }

      canvas.addEventListener('pointerdown', e => {
        dragging=true; moved=false; lastX=e.clientX; lastY=e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!dragging) return;
        const dx=e.clientX-lastX, dy=e.clientY-lastY;
        if (Math.hypot(dx,dy)>1) moved=true;
        yaw += dx*0.007; pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*0.007));
        lastX=e.clientX; lastY=e.clientY;
      });
      canvas.addEventListener('pointerup', e => {
        dragging=false;
        if (moved) return;
        const r=canvas.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
        let best=-1, distance=10;
        for (let i=0;i<projected.length;i++) {
          const d=Math.hypot(x-projected[i][0], y-projected[i][1]);
          if (d<distance) { distance=d; best=i; }
        }
        if (best<0) for (let i=0;i<projected.length-1;i++) {
          const d=pointSegmentDistance(x,y,projected[i],projected[i+1]);
          if (d<distance) { distance=d; best=i+1; }
        }
        if (best>=0) deform(best);
      });
      canvas.addEventListener('wheel', e => {
        e.preventDefault(); zoom=Math.max(1.5,Math.min(15,zoom*Math.exp(-e.deltaY*0.001)));
      }, {passive:false});
      getAction('pause').onclick=() => {
        paused=!paused; getAction('pause').textContent=paused ? 'Play' : 'Pause';
      };
      getAction('reset').onclick=() => {
        embeddingBasePath=normalizePath(embeddingPath(embedding,45));
        pts=embeddingBasePath.map(p=>p.slice()); yaw=-0.55; pitch=0.52; spin=0; zoom=5.8;
        getControl('zoom').value=zoom; getControl('count').value=embeddingBasePath.length;
        getControl('angle').value=45; host.querySelector('[data-readout="angle"]').textContent='45';
      };
      getControl('count').max=embeddingBasePath.length;
      getControl('count').value=embeddingBasePath.length;
      getAction('apply-count').onclick=() => {
        const count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(+getControl('count').value||2)));
        getControl('count').value=count;
        pts=resamplePath(embeddingBasePath,count);
      };
      getAction('apply-angle').onclick=() => {
        const angle=Math.max(1,Math.min(179,+getControl('angle').value||45));
        const count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(+getControl('count').value||embeddingBasePath.length)));
        getControl('angle').value=angle;
        host.querySelector('[data-readout="angle"]').textContent=Number(angle.toFixed(2));
        embeddingBasePath=normalizePath(embeddingPath(embedding,angle));
        pts=count===embeddingBasePath.length ? embeddingBasePath.map(p=>p.slice()) : resamplePath(embeddingBasePath,count);
      };
      for (const action of ['twist-left','twist-right','contract','expand','smooth','jitter'])
        getAction(action).onclick=() => transformShape(action);
      getControl('speed').oninput=e=>rotationSpeed=+e.target.value;
      getControl('strength').oninput=e=>deformStrength=+e.target.value;
      getControl('radius').oninput=e=>deformRadius=+e.target.value;
      getControl('line').oninput=e=>lineWidth=+e.target.value;
      getControl('point').oninput=e=>pointSize=+e.target.value;
      getControl('zoom').oninput=e=>zoom=+e.target.value;
      getControl('line-color').oninput=e=>lineColor=e.target.value;
      getControl('point-color').oninput=e=>pointColor=e.target.value;

      if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe(host);
      else window.addEventListener('resize', resize);
      resize(); draw();
      status.textContent='Running';
      let previous=performance.now();
      function animate(now) {
        if (!paused) spin += (now-previous)*0.00018*rotationSpeed;
        previous=now; draw(); requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
      } catch (error) {
        const host=document.getElementById(ELEMENT_ID);
        const status=host && host.querySelector('[data-readout="status"]');
        if (status) { status.textContent='Error: '+error.message; status.style.color='#ff8b8b'; }
        throw error;
      }
    })();
    """
    js = js.replace("ELEMENT_ID", json.dumps(element_id))
    js = js.replace("POINT_DATA", json.dumps(points.round(7).tolist()))
    js = js.replace("EMBEDDING_DATA", json.dumps(np.asarray(embedding, dtype=float).round(10).tolist()))
    # Run in a self-contained iframe. This isolates JavaScript declarations from
    # Colab and avoids both HTML script sanitization and eval_js bridge errors.
    iframe_document = f"""<!doctype html>
    <html><head><meta charset="utf-8"></head>
    <body style="margin:0;background:#060914;overflow:hidden">
      {body_html}
      <script>{js}</script>
    </body></html>"""
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        shell_name = ""

    if shell_name == "ZMQInteractiveShell":
        # Google Colab, JupyterLab, and VS Code notebooks.
        escaped_document = html_module.escape(iframe_document, quote=True)
        display(HTML(
            f'<iframe srcdoc="{escaped_document}" '
            'style="width:100%;height:720px;border:0;border-radius:14px" '
            'sandbox="allow-scripts"></iframe>'
        ))
    else:
        # Normal Python execution, including VS Code Run/Debug.
        output_path = Path(__file__).with_name("embedding_visualization.html").resolve()
        output_path.write_text(iframe_document, encoding="utf-8")
        print(f"Visualization written to: {output_path}")
        opened = webbrowser.open(output_path.as_uri())
        if not opened:
            print("Open that HTML file in a browser to view the visualization.")


# Running this file/cell displays the example immediately.
show_embedding_3d()
