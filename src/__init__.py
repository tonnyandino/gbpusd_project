"""Project package.

On some networks (security software, school/company filters) HTTPS connections fail
with SSLError because Python's own certificate list does not include the certificate
those filters use. `truststore` makes Python use the operating system's certificate
store instead (the one the web browser uses), which fixes this.
"""
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:            # not installed or not supported: keep Python's defaults
    pass
