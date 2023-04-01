const serveStatic = require('serve-static')
const express = require('express')
const compression = require('compression')

var app = express()

app.use(compression())
app.use(serveStatic(__dirname, {
//   maxAge: '1d',
  setHeaders(res)  {
    // allow caching of .data files
    if (res.req.path.endsWith('.data')) {
      res.setHeader('Cache-Control', 'public, max-age=31536000')
        res.setHeader("Access-Control-Allow-Origin", "*")
        res.setHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
    } else {
        res.setHeader('Cache-Control', 'no-store, max-age=0')
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
    }
  }
}))

app.listen(3000)

// function setCustomCacheControl (res, path) {
//   if (serveStatic.mime.lookup(path) === 'text/html') {
//     // Custom Cache-Control for HTML files
//     res.setHeader('Cache-Control', 'public, max-age=0')
//   }
// }
