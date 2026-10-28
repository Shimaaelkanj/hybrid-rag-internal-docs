# Northwind API Reference

## Authentication
The API uses bearer tokens, which expire after 60 minutes. A new token is obtained by calling POST /v2/auth/refresh with the refresh token. Each API key is limited to 100 requests per minute. Requests over the limit receive HTTP 429 with a Retry-After header.

## Pagination
List endpoints are paginated with the page_size and cursor parameters. The default page_size is 50 and the maximum is 200. Each response contains a next_cursor field, and a null next_cursor means there are no more pages.

## Error Codes
NW-1001 means the token is invalid. NW-1002 means the token has expired and must be refreshed. NW-2004 means the requested resource was not found. NW-5003 means an upstream service timed out; clients should retry with exponential backoff, starting at 1 second, up to a maximum of 5 retries.
