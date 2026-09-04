// Served verbatim -- no placeholder substitution happens at the edge, so
// index.html must be committed fully rendered. If `__APP_VERSION__` or
// `__FORMAT_OPTIONS__` ever reappear in it, production shows a literal
// "v__APP_VERSION__" footer and an empty Format dropdown.
// `python scripts/check_page.py` guards against exactly that.
import html from './index.html';

const GPU_BACKEND = 'https://clearly-gather-deviation-shorter.trycloudflare.com';

// Edge in-memory IP rate limiter (resets when isolate recycles)
const edgeRateStore = new Map(); // "ip|class" -> [timestamps]

// A single flat limit cannot work here: the page polls /status while a track
// renders, so one legitimate visitor makes far more cheap read requests than
// the expensive write requests we actually want to ration. A flat 45/min meant
// a visitor was cut off with a 429 partway through their own generation.
//
// So limit by cost class instead:
//   WRITE - starts or affects GPU work. This is the abuse surface.
//   READ  - status/capacity polling. Cheap, and cutting it off breaks the UI.
const WRITE_ROUTES = ['/generate', '/master/', '/cancel/'];
const LIMITS = { write: 15, read: 300 }; // per IP per minute

function routeClass(pathname) {
  return WRITE_ROUTES.some(r => pathname.startsWith(r)) ? 'write' : 'read';
}

function checkEdgeRateLimit(ip, cls) {
  if (!ip) return true;
  const now = Date.now();
  const windowMs = 60000; // 1 minute
  const key = ip + '|' + cls;

  let timestamps = (edgeRateStore.get(key) || []).filter(t => now - t < windowMs);
  if (timestamps.length >= LIMITS[cls]) {
    edgeRateStore.set(key, timestamps);
    return false;
  }
  timestamps.push(now);
  edgeRateStore.set(key, timestamps);
  return true;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Proxy GPU backend API routes directly to the local RTX 3090 tunnel
    const apiRoutes = ['/generate', '/status/', '/result/', '/gpu', '/cancel/', '/master/', '/quota', '/capacity', '/lounge/'];
    if (apiRoutes.some(route => url.pathname.startsWith(route))) {
      // Handle CORS preflight
      if (request.method === 'OPTIONS') {
        return new Response(null, {
          status: 204,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Max-Age': '86400',
          }
        });
      }

      const clientIp = request.headers.get('CF-Connecting-IP') || request.headers.get('x-real-ip') || '';

      // Edge rate-limit check for aggressive flooding, budgeted per cost class
      if (clientIp && !checkEdgeRateLimit(clientIp, routeClass(url.pathname))) {
        return new Response(JSON.stringify({ detail: 'Edge rate limit exceeded. Please slow down.' }), {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
          }
        });
      }

      const targetUrl = new URL(url.pathname + url.search, GPU_BACKEND);
      const headers = new Headers(request.headers);
      headers.delete('host');
      if (clientIp) {
        headers.set('CF-Connecting-IP', clientIp);
      }

      const proxyInit = {
        method: request.method,
        headers: headers,
        redirect: 'follow'
      };

      if (request.method !== 'GET' && request.method !== 'HEAD') {
        proxyInit.body = request.body;
      }

      try {
        const response = await fetch(targetUrl.toString(), proxyInit);
        const respHeaders = new Headers(response.headers);
        respHeaders.set('Access-Control-Allow-Origin', '*');
        respHeaders.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        respHeaders.set('Access-Control-Allow-Headers', '*');
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: respHeaders
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: 'GPU tunnel unreachable: ' + err.message }), {
          status: 502,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
          }
        });
      }
    }

    if (url.pathname === '/health') {
      return new Response('OK', { status: 200 });
    }

    return new Response(html, {
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'public, max-age=0, must-revalidate',
        'Access-Control-Allow-Origin': '*'
      }
    });
  }
};
