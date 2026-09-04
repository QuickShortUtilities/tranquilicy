// Served verbatim -- no placeholder substitution happens at the edge, so
// index.html must be committed fully rendered. If `__APP_VERSION__` or
// `__FORMAT_OPTIONS__` ever reappear in it, production shows a literal
// "v__APP_VERSION__" footer and an empty Format dropdown.
// `python scripts/check_page.py` guards against exactly that.
import html from './index.html';

const GPU_BACKEND = 'https://clearly-gather-deviation-shorter.trycloudflare.com';

// Edge in-memory IP rate limiter (resets when isolate recycles)
const edgeRateStore = new Map(); // ip -> [timestamps]

function checkEdgeRateLimit(ip) {
  if (!ip) return true;
  const now = Date.now();
  const windowMs = 60000; // 1 minute
  const maxCallsPerMin = 45; // allows frequent status polling but blocks abusive flooding

  let timestamps = edgeRateStore.get(ip) || [];
  timestamps = timestamps.filter(t => now - t < windowMs);
  if (timestamps.length >= maxCallsPerMin) {
    return false;
  }
  timestamps.push(now);
  edgeRateStore.set(ip, timestamps);
  return true;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Proxy GPU backend API routes directly to the local RTX 3090 tunnel
    const apiRoutes = ['/generate', '/status/', '/result/', '/gpu', '/cancel/', '/master/', '/quota'];
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

      // Edge rate-limit check for aggressive flooding
      if (clientIp && !checkEdgeRateLimit(clientIp)) {
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
