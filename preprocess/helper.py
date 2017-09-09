import pandas as pd

def filter_df_base_on_minfb(df, target, min_fb):
    """"""
    count_df = df.groupby(target).size()
    count = zip(
        count_df.index.tolist(),
        count_df.tolist()
    )

    targets = set(
        map(
            lambda x:x[0],
            filter(
                lambda y:y[1] > min_fb,
                count
            )
        )
    )

    # now filter out triplet based on user
    i = df.columns.tolist().index(target)
    return pd.DataFrame(
        map(
            lambda y:(y[0], y[1], y[2]),
            filter(
                lambda x:x[i] in targets,
                df.as_matrix()
            )
        ),
        columns=df.columns
    )

def densify_trp_df(df, min_fb=5):
    """"""
    if isinstance(min_fb, int):
        min_fb_tup = (min_fb, min_fb)
    elif isinstance(min_fb, Iterable):
        if len(min_fb) > 2:
            raise Exception
        min_fb_tup = min_fb

    cols = df.columns

    filt_diff = 1
    df_cur = df
    while filt_diff > 0:
        df_filt_col0 = filter_df_base_on_minfb(df_cur, cols[0], min_fb[0])
        df_filt_col1 = filter_df_base_on_minfb(df_filt_col0, cols[1],
                                               min_fb[1])
        filt_diff = len(df_cur) - len(df_filt_col1)
        df_cur = df_filt_col1

    return df_cur
