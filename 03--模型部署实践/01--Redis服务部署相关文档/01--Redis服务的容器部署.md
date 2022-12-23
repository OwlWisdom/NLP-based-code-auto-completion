# 01--Redis服务的容器部署

<span style='color:brown'>**参考链接：**</span>

- Docker hub -- [redis official image](https://hub.docker.com/_/redis)
- Docker hub -- [redis / redis-stack](https://hub.docker.com/r/redis/redis-stack)
- Github -- [flask-restful-example](https://github.com/qzq1111/flask-restful-example)







## How to use this image

### start a redis instance

```shell
$ docker run --name some-redis -d redis
```

### start with persistent storage

```shell
$ docker run --name some-redis -d redis redis-server --save 60 1 --loglevel warning
```

There are several different persistence strategies to choose from. This one will save a snapshot of the DB every 60 seconds if at least 1 write operation was performed (it will also lead to more logs, so the `loglevel` option may be desirable). If persistence is enabled, data is stored in the `VOLUME /data`, which can be used with `--volumes-from some-volume-container` or `-v /docker/host/dir:/data` (see [docs.docker volumes](https://docs.docker.com/engine/tutorials/dockervolumes/)).

For more about Redis Persistence, see http://redis.io/topics/persistence.

### connecting via `redis-cli`

```shell
$ docker run -it --network some-network --rm redis redis-cli -h some-redis
```



## Run Redis Stack on Docker

### "How to install Redis Stack using Docker"

To get started with Redis Stack using Docker, you first need to select a Docker image:

- `redis/redis-stack` contains both Redis Stack server and RedisInsight. This container is best for local development because you can use RedisInsight to visualize your data.
- `redis/redis-stack-server` provides Redis Stack but excludes RedisInsight. This container is best for production deployment.

### Getting started

To start Redis Stack server using the `redis-stack` image, run the following command in your terminal:

```shell
$ docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

You can then connect to the server using `redis-cli`, just as you connect to any Redis instance.

If you don’t have `redis-cli` installed locally, you can run it from the Docker container:

```shell
$ docker exec -it redis-stack redis-cli
```

#### RedisInsight

The `docker run` command above also exposes RedisInsight on port 8001. You can use RedisInsight by pointing your browser to [http://localhost:8001](http://localhost:8001/).

### Configuration

### Persistence

To persist your Redis data to a local path, specify `-v` to configure a local volume. This command stores all data in the local directory `local-data`:

```shell
$ docker run -v /local-data/:/data redis/redis-stack:latest
```

### Ports

If you want to expose Redis Stack server or RedisInsight on a different port, update the left hand of portion of the `-p` argument. This command exposes Redis Stack server on port `10001` and RedisInsight on port `13333`:

```shell
$ docker run -p 10001:6379 -p 13333:8001 redis/redis-stack:latest
```

### Config files

By default, the Redis Stack Docker containers use internal configuration files for Redis. To start Redis with local configuration file, you can use the `-v` volume options:

```shell
$ docker run -v `pwd`/local-redis-stack.conf:/redis-stack.conf -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### Environment variables

To pass in arbitrary configuration changes, you can set any of these environment variables:

- `REDIS_ARGS`: extra arguments for Redis
- `REDISEARCH_ARGS`: arguments for RediSearch
- `REDISJSON_ARGS`: arguments for RedisJSON
- `REDISGRAPH_ARGS`: arguments for RedisGraph
- `REDISTIMESERIES_ARGS`: arguments for RedisTimeSeries
- `REDISBLOOM_ARGS`: arguments for RedisBloom

For example, here's how to use the `REDIS_ARGS` environment variable to pass the `requirepass` directive to Redis:

```shell
$ docker run -e REDIS_ARGS="--requirepass redis-stack" redis/redis-stack:latest
```

Here's how to set a retention policy for RedisTimeSeries:

```shell
$ docker run -e REDISTIMESERIES_ARGS="RETENTION_POLICY=20" redis/redis-stack:latest
```



