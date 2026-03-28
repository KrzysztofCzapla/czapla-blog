---
title: Security in FastAPI
description: Most important aspects of the FastAPI security and their practical implementations
date: 2026-03-28
---

# Intro

Many people often overlook security in FastAPI. Despite it coming without the `batteries included™`, it's not really
complicated to create a secure microservice quickly. This article is a quick check-list of things to implement to sleep
better at night.

Note:
`This guide is not a step-by-step booklet on what to do exactly, but just a "checklist" of things you need to keep in mind with examples.`

# Authentication

## CORS

CORS (Cross-Origin Resource Sharing) is a mechanism/way to reject requests from sites that do not share your domain.
It also has a few other options, for example you can specify the HTTP methods to allow/reject.

[In FastAPI it comes already implemented](https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware), 
you just need to add the middleware:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["example.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
```

Now, technically if someone sends a request from another site to your site, the browser should reject it, right?

Unfortunately, it's not that simple. While the attacker won't get access to the response, the request will still get
to the service and execute all the logic. So, for example, if an attacker sends POST request to your endpoint
that's gonna delete a resource, the resource will be deleted, just no response will be sent back.
This will be mitigated with using `SameSite=Strict` policy on the JWT cookies, but more on that later.

It also does nothing to protect from requests sent from outside the browser.

## JWT

JWT (JSON Web Tokens) is a standard for transmitting information securely.

We will store 2 cookies in the domain of our service:
- Access Token - 15 minutes lifetime - it will be used to access the endpoints. It is short-lived so if someone steals it, we won't have a problem
- Refresh Token - X days lifetime - used to get new access tokens. We will rotate this one each time we generate a new access token,
so if someone steals it and both attacker and the user will try to use it, we will know it was compromised.

This is not built-in in FastAPI, but there are nice libraries that we can use. The tokens are generated using a secret key.

The libraries we use, except the obvious ones:
```python-jose passlib[bcrypt]```

Full example in one file:
```python
from fastapi import FastAPI, Depends, HTTPException, Response, Cookie
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from sqlalchemy import create_engine, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, Session
import uuid

# --- config ---
SECRET = "super-secret"
ALGO = "HS256"
ACCESS_EXPIRE_MIN = 15
REFRESH_EXPIRE_DAYS = 7

# --- auth utils stuff ---
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access(user_id: str):
    return jwt.encode({
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRE_MIN),
        "type": "access"
    }, SECRET, algorithm=ALGO)

def create_refresh(user_id: str):
    jti = str(uuid.uuid4())
    token = jwt.encode({
        "sub": user_id,
        "jti": jti,
        "exp": datetime.utcnow() + timedelta(days=REFRESH_EXPIRE_DAYS),
        "type": "refresh"
    }, SECRET, algorithm=ALGO)
    return token, jti

# --- main logic ---
app = FastAPI()


@app.post("/login")
def login(response: Response, body: LoginRequestBody, db: Session = Depends(get_db)):
    # first verify the password and everything here of course
    user_id = body["id"]
    access = create_access(user_id)
    refresh, jti = create_refresh(user_id)
    db.add(RefreshToken(
        jti=jti,
        user_id=user_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS),
        used=False
    ))
    db.commit()

    response.set_cookie("access_token", access, httponly=True, secure=True, samesite="strict")
    response.set_cookie("refresh_token", refresh, httponly=True, secure=True, samesite="strict")

    return {"msg": "logged in"}


@app.post("/refresh")
def refresh(response: Response, refresh_token: str = Cookie(None), db: Session = Depends(get_db)):
    if not refresh_token:
        raise HTTPException(401)
    try:
        payload = jwt.decode(refresh_token, SECRET, algorithms=[ALGO])
        jti = payload["jti"]
        user_id = payload["sub"]
    except JWTError:
        raise HTTPException(401)

    token_db = db.get(RefreshToken, jti)
    if not token_db or token_db.used or token_db.expires_at < datetime.utcnow():
        # Here we could implement more logic regarding the detection. Maybe logging/alerting.
        raise HTTPException(401, "reuse detected or invalid")

    # rotate
    token_db.used = True
    new_refresh, new_jti = create_refresh(user_id)
    db.add(RefreshToken(
        jti=new_jti,
        user_id=user_id,
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_EXPIRE_DAYS),
        used=False
    ))
    db.commit()

    new_access = create_access(user_id)
    response.set_cookie("access_token", new_access, httponly=True, secure=True, samesite="strict")
    response.set_cookie("refresh_token", new_refresh, httponly=True, secure=True, samesite="strict")
    return {"msg": "refreshed"}


# Example endpoint "protected" endpoint
@app.get("/me")
def me(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(401)

    try:
        payload = jwt.decode(access_token, SECRET, algorithms=[ALGO])
        return {"user_id": payload["sub"]}
    except JWTError:
        raise HTTPException(401)
```

Important thing here is:
```python
response.set_cookie("access_token", new_access, httponly=True, secure=True, samesite="strict")
response.set_cookie("refresh_token", new_refresh, httponly=True, secure=True, samesite="strict")
```
We NEED to set the `httponly=True` and `samesite="strict"`, so we are saved from CSRF and XSS attacks.

BTW - treat the above code as a mocky way to showcase the general logic. Every service is different so this doesn't include
actual verification of the user or usage of modern python standard (async, not-using SQLite in PROD etc).

This approach assumes the client is on the same domain as the backend. If that's not the case you need to lower the strictness
of JWT cookies and implement CRSF tokens. You can use a package like [fastapi-csrf-protect](https://pypi.org/project/fastapi-csrf-protect/).


# DB

## Password hashing

To keep the DB safe - don't store passwords as plain text. You must hash them. That way even if someone gets access to your DB,
no password will be leaked.

We can utilize the previously used `passlib` library.

A short example would be:
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

password = "ilovehotelsbecauseihatemylife"
hashed = pwd_context.hash(password)

is_valid = pwd_context.verify("ilovehotelsbecauseihatemylife", hashed)
```

Everything happens automatically here, you don't need to specify salt or a secret key.

## Maybe don't use password?

Storing password comes with an overhead. You may want to completely opt-out of storing password of users.
Even strong hashing mechanism can be broken.

There are 2 non-conflicting ways to do skip storing passwords:
- Let users login/register via email - Very straightforward, instead of requiring a password when logging in, just send
an email to the user with a link that will generate new JWT cookies
- Use OAuth2 - This will allow users to login using their accounts from external providers like Google.

In my projects I often use both to allow users the maximum freedom. If you integrate your JWT cookies with OAuth2 and emails
well then everything goes very smoothly.

## SQL Injection

SQL injection is a situation in which an attacher `injects` their own SQL code/command into your query. Example:
```python
@app.get("/users")
def get_user(username: str = Query(...)):
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()

    # Vulnerable to SQL Injection
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

    result = cursor.fetchall()
    conn.close()
    return {"data": result}
```

If the `username` query equals `' OR '1'='1` then the attacker would get all the users data.

Of course this is a very naive example.

To avoid this use an ORM (like SQLAlchemy or Tortoise-ORM) or parameterized queries.

## Network

A good idea might be to not expose your DB to the internet at all. If you keep your DB in the same server as your service,
you might only expose it internally. That way the service will be able to use it, but no one will be able to connect
from the outside.

Of course, usually microservices are separated from the DB server, so they can scale horizontally.

# DDOS protection/Rate limiting

To protect from attacks that send a lots of requests at the same time you might want to implement limits of how many times a user/ip adress
can send a request per minute/hour.

Short example:
```python
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

@app.get("/")
@limiter.limit("5/minute")
def home(request: Request):
    return {"ok": True}
```

```python
limiter = Limiter(key_func=get_remote_address)
```
the `key_func` here can be overriden to for example limit based on a user and not an IP adress.

Of course this is never enough to protect yourelf from a DDOS attack. to actually protect from that
you will probably need an external provider's help. 

You can try to implement a cloudflare verification protection or use a CDN.
In the most popular cloud providers there are some tools that use machine learning to detect those kinds of requests
and they will block them before they can get to your app.

# Miscellaneous

Other important aspects:
- Never store secrets in the code. No matter if it's only for development or internal, always use a SecretManager or at least
an `.env` file.
- Always use HTTPS. You can easily set that up. For example using letsencrypt + nginx. If you use docker/docker-compose
you can implement this using just docker-compose file and specifying the correct images.

# Outro

That's pretty much it, there are not that many things you need to watch out for actually if you count them.

Thanks for reading.