import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ILREC_ENV_CONFIG = {
    "AmazonEnv-v0": {
        "dataset": "amazon",
        "leave_threshold": 15,
        "env_class": "AmazonEnv",
    },
    "SteamEnv-v0": {
        "dataset": "steam",
        "leave_threshold": 50,
        "env_class": "SteamEnv",
    },
}
ILREC_DATA_ROOT = Path(
    os.environ.get(
        "ILREC_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "environments" / "ILRec" / "data",
    )
)
ILREC_SOURCE_ROOT = Path(os.environ.get("ILREC_SOURCE_ROOT", os.environ.get("ILREC_ROOT", "/home/hehui/il-rec")))


def _get_ilrec_dataset(env):
    return ILREC_ENV_CONFIG[env]["dataset"]


def _read_ilrec_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required ILRec table is missing: {path}")
    return pd.read_csv(path)


def _load_ilrec_tables(env, split):
    dataset = _get_ilrec_dataset(env)
    dataset_root = ILREC_DATA_ROOT / dataset
    df_data = _read_ilrec_csv(dataset_root / f"{split}.csv")
    df_user = _read_ilrec_csv(dataset_root / "user.csv").set_index("user_id")
    df_item = _read_ilrec_csv(dataset_root / "item.csv").set_index("item_id")
    for column in ("user_id", "item_id", "timestamp"):
        if column in df_data:
            df_data[column] = df_data[column].astype(int)
    if "rating" in df_data:
        df_data["rating"] = df_data["rating"].astype(float)
    df_user.index = df_user.index.astype(int)
    df_item.index = df_item.index.astype(int)
    return df_data, df_user, df_item, []


def _load_ilrec_matrix(dataset, split, read_user_num=None):
    from environments.ILRec.env.ILRecEnv import ILRecEnv

    env_root = ILREC_SOURCE_ROOT / "env" / dataset
    mat_path = env_root / f"{dataset}_{split}.npy"
    distance_path = env_root / f"{split}_distance_mat.pickle"
    return ILRecEnv.load_resources(mat_path, distance_path, read_user_num=read_user_num)


def get_features(env, is_userinfo=False):
    if env == "CoatEnv-v0":
        user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        reward_features = ["rating"]
    elif env == "KuaiRand-v0":
        user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author',
                         'follow_user_num_range',
                         'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                        + [f'onehot_feat{x}' for x in range(18)]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"]
        reward_features = ["is_click"]
    elif env == "KuaiEnv-v0":
        user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(4)] + ["duration_normed"]
        reward_features = ["watch_ratio_normed"]
    elif env == "YahooEnv-v0":
        user_features = ["user_id"]
        item_features = ['item_id']
        reward_features = ["rating"]
    elif env in ILREC_ENV_CONFIG:
        user_features = ["user_id"]
        item_features = ["item_id"]
        reward_features = ["rating"]

    return user_features, item_features, reward_features

def get_training_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_train, df_user, df_item, list_feat = CoatEnv.get_df_coat("train.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_train, df_user, df_item, list_feat = KuaiRandEnv.get_df_kuairand("train_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_train, df_user, df_item, list_feat = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_train, df_user, df_item, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")
    elif env in ILREC_ENV_CONFIG:
        df_train, df_user, df_item, list_feat = _load_ilrec_tables(env, "train")

    return df_train, df_user, df_item, list_feat

def get_training_item_domination(env):
    item_feat_domination = None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        item_feat_domination = CoatEnv.get_domination()
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        item_feat_domination = KuaiRandEnv.get_domination()
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        item_feat_domination = KuaiEnv.get_domination()
    elif env == "YahooEnv-v0":
        item_feat_domination = None
    elif env in ILREC_ENV_CONFIG:
        item_feat_domination = None

    return item_feat_domination


def get_val_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_val, df_user_val, df_item_val, list_feat = CoatEnv.get_df_coat("test.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiRandEnv.get_df_kuairand("test_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_val, df_user_val, df_item_val, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")
    elif env in ILREC_ENV_CONFIG:
        df_val, df_user_val, df_item_val, list_feat = _load_ilrec_tables(env, "test")

    return df_val, df_user_val, df_item_val, list_feat


def get_common_args(args):
    env = args.env

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')

    parser.add_argument('--is_need_transform', dest='need_transform', action='store_true')
    parser.add_argument('--no_need_transform', dest='need_transform', action='store_false')


    if env == "CoatEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=10, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "YahooEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiRand-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=1)
        parser.add_argument("--yfeat", type=str, default="is_click")

        parser.add_argument('--leave_threshold', default=0, type=float)
        parser.add_argument('--num_leave_compute', default=10, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiEnv-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=True)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--yfeat", type=str, default="watch_ratio_normed")

        # parser.add_argument('--leave_threshold', default=1, type=float)
        # parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--leave_threshold', default=0, type=float)
        parser.add_argument('--num_leave_compute', default=10, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env in ILREC_ENV_CONFIG:
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=False)
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=ILREC_ENV_CONFIG[env]["leave_threshold"], type=float)
        parser.add_argument('--num_leave_compute', default=4, type=int)
        parser.add_argument('--max_turn', default=100, type=int)

    parser.add_argument('--force_length', type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)
    if env == "KuaiEnv-v0":
        args.use_userEmbedding = False

    return args


def get_true_env(args, read_user_num=None):
    if args.env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        mat, df_item, mat_distance = CoatEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
        env = CoatEnv(**kwargs_um)
        env_task_class = CoatEnv
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        mat, mat_distance = YahooEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}

        env = YahooEnv(**kwargs_um)
        env_task_class = YahooEnv
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        mat, list_feat, mat_distance = KuaiRandEnv.load_mat(args.yfeat, read_user_num=read_user_num)
        kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "mat_distance": mat_distance,
                     "list_feat": list_feat,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
        env = KuaiRandEnv(**kwargs_um)
        env_task_class = KuaiRandEnv
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        mat, lbe_user, lbe_item, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "list_feat": list_feat,
                     "df_video_env": df_video_env,
                     "df_dist_small": df_dist_small}
        env = KuaiEnv(**kwargs_um)
        env_task_class = KuaiEnv
    elif args.env in ILREC_ENV_CONFIG:
        from environments.ILRec.env.ILRecEnv import AmazonEnv, SteamEnv
        dataset = _get_ilrec_dataset(args.env)
        env_task_class = AmazonEnv if args.env == "AmazonEnv-v0" else SteamEnv
        mat, mat_distance = _load_ilrec_matrix(dataset, "test", read_user_num=read_user_num)
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
        env = env_task_class(**kwargs_um)
    return env, env_task_class, kwargs_um # simulated环境，环境类别指示，参数
