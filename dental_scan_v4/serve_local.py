from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
print("Serving Dental Scan V4 at http://localhost:8000")
print("Phone camera testing generally requires HTTPS; localhost works on the same computer.")
ThreadingHTTPServer(("0.0.0.0",8000),SimpleHTTPRequestHandler).serve_forever()
