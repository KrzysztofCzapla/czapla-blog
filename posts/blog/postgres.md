---
title: PostgreSQL setup for simple python projects
description: How to setup PostgreSQL with Docker and PGBouncer
date: 2026-04-11
---

# Intro

This is a guide on how to set up PostgreSQL using Docker (with Compose) and PGBouncer. I assume you know basic Docker and PostgreSQL.

The setup is for a framework-agnostic python code, I personally use it mostly with FastAPI apps.
If you want to use it with Django, you still can, but keep in mind that Django has some our functionalities already built-in, for example extension handling or migrations.

**Note**: To keep this article short, we will skip the very basic stuff like installing python packages or storing secrets in a safe way.
So, if you see any passwords stored as plain text in commitable files here, do not panic.

Our setup has the following benefits:

1. **Connection Pooler** - we don't have to worry about the number of concurrent connections, this will take care of it
3. **Automatic extension download and triggers setup** - If we don't use something like Django, this will come in handy
4. **Small and easy to change**

Also let's list tools that we will use:

1. **PostgreSQL & Python**, duh
2. **PGBouncer** - connection pooler/load balancer 
3. **Docker with Compose** - for creating containers and images
4. **DBMate** - lightweight migrations manager
5. **PugSQL** - Python interface to connect to the database, under the hood it uses SQLAlchemy connection engine


# PostgreSQL

First, lets create the DB server itself. To do that create `docker-compose.yaml` file like that:
```yaml
services:
  postgres:
    container_name: postgres
    image: postgres:16
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres_user
      POSTGRES_PASSWORD: postgres_pass
      POSTGRES_DB: main
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/db_init:/docker-entrypoint-initdb.d
      - ./db/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./db/backus/:/backups/
    command: [ "postgres", "-c", "config_file=/etc/postgresql/postgresql.conf" ]

volumes:
  pgdata:
```

I think everything here is pretty much standard Docker practices. Of course we should be taking the postgres password from a secrets file, but again, we will skip it to keep this post short.

Also, as you can see, looking at the volumes, we will have a dedicated `db` folder to keep our configuration and initialization files.

We also override the start command:
```yaml
command: [ "postgres", "-c", "config_file=/etc/postgresql/postgresql.conf" ]
```
We do that to specify which config file postgres should use when starting, so we can change it.

Now, let's create the `db` folder and inside `db_init` folder. Let's also create `db/postgresql.conf` file:
```
listen_addresses = '*'
max_connections=100
```

There is a long list of things we can specify here, let's keep it short for now. But it's nice to have it so you can easily change anything you want later.

## (Optional) Installation of extensions and other scripts on initialization

If you'd like to install any extension, you can do it in the `db_init` folder. All the files here will run on first database initialization.

Let's create `extensions.sql`:
```
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

This will auto-install `pg_stat_statements`, which is used for storing statistics of our sql queries and other stuff.

You can do the same for any custom functions you would like to create automatically, `triggers.sql`:
```sql
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

# PGBouncer

Now, having this setup we could already connect to our DB, but we want to put a connection pooler between our services and the DB server.

Why? Imagine that we have maximum of 100 concurrent connections but our services create 120 connections at the same time.
Some of them would fail, which is far from ideal.

To mitigate this issue we can add a connection pooler, which will keep the extra connections in a "queue" and eventually it will connect them
to the DB server so they will not fail.

To add PGBouncer, let's add its container to the `docker-compose.yaml`:
```yaml
services:
    postgres:
      ...
    
    pgbouncer:
        container_name: pgbouncer
        image: edoburu/pgbouncer:v1.25.1-p0
        restart: always
        volumes:
          - ./db/pgbouncer:/etc/pgbouncer
        ports:
          - "6432:6432"
```

As you can see we will also add some setups files to the `db` folder.

Regarding the ports - we will use other ports to distinct this from the normal postgres server.

Let's create `db/pgbouncer/pgbouncer.ini`:
```
[databases]
main = host=postgres port=5432 dbname=main user=postgres_user password=postgres_pass

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 500
default_pool_size = 100
reserve_pool_size = 20
reserve_pool_timeout = 5
server_idle_timeout = 30
client_login_timeout = 10
query_timeout = 0
admin_users = postgres_user
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid
```

In the first section we specify how to connect to the DB and in the second one we actually specify the PGBouncer settings.

As you can see we wll allow maximum of 500 concurrent connections, while the DB itself only accepts 100. Some clients will wait for a pool slot until it is available.

Let's also create the userlist.txt:
```
"postgres_user" "postgres_pass"
```

# Connecting via Python

First we need to create a Dockerfile and add it to the docker-compose file. I assume you know how to do this.

### PugSQL

[PugSQL](https://pugsql.org/) is a nice library that allows us to write queries in SQL files.

First we need to specify the folder with our queries and then connect to the DB.

Once we have our app running, we can connect it to the DB using this connection string, `db.py`:
```python
from pathlib import Path

import pugsql

queries = pugsql.module(Path(__file__).parent / "queries/")
queries.connect('postgresql://postgres_user:postgres_pass@pgbouncer:6432/main')
```

As you can see we are connecting to the pgbouncer on port 6432, we are directly using `pgbouncer` as the URL because
in docker compose you can specify URL by a container's name.

To execute any queries, you first need to create one in the `queries` folder, `first.sql`:
```sql
-- :name simple :scalar
SELECT 1;
```

You specify the name of the query using SQL comments.

And then in python:
```python
from app.db import queries

print(queries.simple())
>> 1
```

### Migrations

AS you can see we haven't even created any tables in our DB. To do that we can use DBMate. We will use the Docker version so we dont have to install anything locally.

We first will write manual migrations and then DBMate will execute them. First we need to create migration file by executing:
```docker run --rm -v $(pwd)/db:/db ghcr.io/amacneil/dbmate new create_users_table```

Now `migrations` folder should be created with a new file like: `1323123131_create_users_table.sql`


Let's write a basic migration in this file:
```sql
CREATE TABLE basic_table (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    name TEXT NOT NULL,
    CONSTRAINT basic_table_pk PRIMARY KEY (id)
);
```

And now let's apply:
```
docker run --rm \
  -v $(pwd)/db:/db \
  -w /db \
  --network <your docker compose network here > \
  -e DATABASE_URL="postgres://postgres_user:postgres_pass@postgres:5432/main?sslmode=disable" \
  ghcr.io/amacneil/dbmate \
  up
```

And that's how we do changes to the DB in this approach. You can read more about it on DBMate's github page.

### Backups

This is tricky and depends mostly on your general infrastructure. There are many ways to do this.

The easiest one, assuming you have one VPS running everything would be creating a cron job like this:
`0 2 * * * docker exec postgres pg_dump -U postgres_user main > /backups/appdb.sql`

There are many ways here, including creating a docker compose container that will only do this, or even creating a scheduled task in your app but cron job seems the simplest.

# Outro

I've used this setup before on my projects, and it worked well for my use-cases.

Thanks for reading. 