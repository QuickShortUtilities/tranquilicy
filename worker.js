import html from './index.html';

const GPU_BACKEND = 'https://clearly-gather-deviation-shorter.trycloudflare.com';

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Proxy GPU backend API routes directly to the local RTX 3090 tunnel
    const apiRoutes = ['/generate', '/status/', '/result/', '/gpu', '/cancel/', '/master/'];
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

      const targetUrl = new URL(url.pathname + url.search, GPU_BACKEND);
      const headers = new Headers(request.headers);
      headers.delete('host');

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
