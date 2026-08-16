"""Single source of truth for Multi-DocID tokenizer and route token IDs."""


def build_multi_view_special_tokens(
    n_levels: int, n_codes: int, n_views: int
):
    if n_levels <= 0 or n_codes <= 0 or n_views <= 0:
        raise ValueError("n_levels, n_codes, and n_views must be positive")
    if n_levels % n_views != 0:
        raise ValueError("RQ levels must divide evenly across views")
    view_tokens = [f"<view_{view}>" for view in range(n_views)]
    rq_tokens = [
        f"<r{level}_{code}>"
        for level in range(n_levels)
        for code in range(n_codes)
    ]
    return view_tokens + rq_tokens


def add_multi_view_special_tokens(tokenizer, tokens):
    return tokenizer.add_tokens(list(tokens), special_tokens=True)


def validate_atomic_tokens(tokenizer, tokens):
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if token_id == tokenizer.unk_token_id or encoded != [token_id]:
            raise ValueError(
                f"special token {token!r} is not atomic: id={token_id}, encoded={encoded}"
            )


def route_token_ids(tokenizer, route: str, add_eos: bool):
    token_ids = tokenizer.encode(route, add_special_tokens=False)
    route_tokens = route.split()
    if len(token_ids) != len(route_tokens):
        raise ValueError(
            f"route is not one-token-per-codeword: {route!r} -> {token_ids}"
        )
    if add_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        token_ids = token_ids + [tokenizer.eos_token_id]
    return token_ids
