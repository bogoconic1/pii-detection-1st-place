from spacy.lang.en import English
import re
import pandas as pd
import json
import dateutil

nlp = English()

subtitles = [
    "Dr",
    "Mr",
    "Ms",
    "Mrs",
    "By",
    "Dr.",
    "Dr .",
    "Mr.",
    "Mr .",
    "Ms.",
    "Ms .",
    "Mrs.",
    "Mrs .",
    "By.",
    "By .",
]


def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue

    return spans


def regex_predictions(data):

    print("adding regex predictions")

    email_regex = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
    id_num_regex = re.compile("[\w\.\:\-\_\|]*\d{6,}")
    emails = []
    phone_nums = []
    id_nums = []

    for _data in data:
        # email
        for token_idx, token in enumerate(_data["tokens"]):
            if re.fullmatch(email_regex, token) is not None:
                emails.append(
                    {
                        "document": _data["document"],
                        "token": token_idx,
                        "label": "B-EMAIL",
                        "token_str": token,
                    }
                )
        # phone number
        matches = phone_num_regex.findall(_data["full_text"])
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, _data["tokens"])
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append(
                    {
                        "document": _data["document"],
                        "token": token_idx,
                        "label": f"{prefix}-PHONE_NUM",
                        "token_str": _data["tokens"][token_idx],
                    }
                )

        for token_idx, token in enumerate(_data["tokens"]):
            match = id_num_regex.match(token)
            if match is None:
                continue
            id_nums.append(
                {
                    "document": _data["document"],
                    "token": token_idx,
                    "label": "B-ID_NUM",
                    "token_str": token,
                }
            )

    return pd.DataFrame(emails + phone_nums + id_nums)


# ================================================================================================
def filter_student_preds(row):
    if not "NAME_STUDENT" in row["label"]:
        return True
    else:
        try:
            if (
                (
                    row["token_str"].istitle()
                    or row["token_str"] == "\n"
                    or row["token_str"] == "-"
                )
                and (not any(x.isdigit() for x in row["token_str"]))
                and (not any(x == "_" for x in row["token_str"]))
            ):
                return True
            else:
                return False
        except:
            return False


def is_date_or_time(row):
    try:
        _ = dateutil.parser.parse(row["token_str"])
        return True
    except:
        return False


def is_valid_id_num(row):

    if not "ID_NUM" in row["label"]:
        return True
    else:
        if len(row["token_str"]) < 4:
            return False
        if is_date_or_time(row["token_str"]):
            return False
        return True


# ================================================================================================
def postprocess_id_phone(df, DEBUG=False):

    print("postprocessing id and phone")

    sub = df

    digit_pat = r"^\d+$"
    phone_dot_pat = r"^\d{3}\.\d{3}\.\d{4}$"
    id_dot_pat = r"^\d{3}\.\d{4}\.\d{4}$"
    all_dot_pat = r"\d+\.\d+\.\d+"

    ssn_id_num_pat = r"^\d{3}-\d{2}-\d{4}$"
    phone_hyphen_pat = r"^\d{3}-\d{3}-\d{4}$"

    id_comma_pat = r"^\d{1,2}\,\d{1,2}\,\d{1,2},\d{1,2}$"
    alphabet_pattern = r"[a-zA-Z]"

    for i in range(len(sub)):

        # ========================================================================

        string_to_check = sub.token_str[i]
        if DEBUG:
            old_label = sub.label[i]

        if 1 + 1 == 2:  # "ID_NUM" in sub.label[i] or "PHONE_NUM" in sub.label[i]:

            try:
                if re.match(digit_pat, string_to_check):

                    if len(string_to_check) >= 9 and "PHONE_NUM" in sub.label[i]:
                        sub.label[i] = "B-ID_NUM"

                        if DEBUG:
                            if old_label != sub.label[i]:
                                print(string_to_check, old_label, sub.label[i])

                        continue
            except:
                pass

            try:
                if re.match(all_dot_pat, string_to_check):
                    if re.match(phone_dot_pat, string_to_check):
                        sub.label[i] = "B-PHONE_NUM"
                    else:
                        if "x" in string_to_check:
                            sub.label[i] = "B-PHONE_NUM"
                        elif re.match(id_dot_pat, string_to_check):
                            sub.label[i] = "B-ID_NUM"

                    if DEBUG:
                        if old_label != sub.label[i]:
                            print(string_to_check, old_label, sub.label[i])

                    continue
            except:
                pass

            try:
                if re.match(id_comma_pat, string_to_check):
                    sub.label[i] = "B-ID_NUM"
                    if DEBUG:
                        if old_label != sub.label[i]:
                            print(string_to_check, old_label, sub.label[i])
                    continue
            except:
                pass

            try:
                if "PHONE_NUM" in sub.label[i] and re.search(
                    alphabet_pattern, string_to_check
                ):
                    if (
                        "x" not in string_to_check
                        and "X" not in string_to_check
                        and "Ext" not in string_to_check
                        and "ext" not in string_to_check
                        and "EXT" not in string_to_check
                    ):
                        sub.label[i] = "B-ID_NUM"
                        if DEBUG:
                            if old_label != sub.label[i]:
                                print(string_to_check, old_label, sub.label[i])
                        continue
            except:
                pass

            # ========================================================================

            string_to_check = ""

            if i + 4 < len(sub):
                # if it is not the first index of a contiguous segment, or it is not the last index of a contiguous segment, skip it
                if (
                    i - 1 >= 0
                    and sub.document[i - 1] == sub.document[i]
                    and sub.token[i - 1] + 1 == sub.token[i]
                ) or (
                    i + 5 < len(sub)
                    and sub.document[i + 5] == sub.document[i]
                    and sub.token[i + 5] - 5 == sub.token[i]
                ):
                    pass

                else:
                    if (
                        len(
                            set(
                                [
                                    sub.document[i],
                                    sub.document[i + 1],
                                    sub.document[i + 2],
                                    sub.document[i + 3],
                                    sub.document[i + 4],
                                ]
                            )
                        )
                        == 1
                        and sub.token[i] + 1 == sub.token[i + 1]
                        and sub.token[i] + 2 == sub.token[i + 2]
                        and sub.token[i] + 3 == sub.token[i + 3]
                        and sub.token[i] + 4 == sub.token[i + 4]
                    ):
                        for inner_index in range(i, i + 5):
                            string_to_check += sub.token_str[inner_index]

            try:
                if re.match(ssn_id_num_pat, string_to_check):
                    for inner_index in range(i, i + 5):
                        old_label = sub.label[inner_index]
                        if inner_index == i:
                            sub.label[inner_index] = "B-ID_NUM"
                        else:
                            sub.label[inner_index] = "I-ID_NUM"

                        if DEBUG:
                            if old_label != sub.label[inner_index]:
                                print(
                                    string_to_check, old_label, sub.label[inner_index]
                                )

                    continue

                elif re.match(phone_hyphen_pat, string_to_check):
                    for inner_index in range(i, i + 5):
                        old_label = sub.label[inner_index]
                        if inner_index == i:
                            sub.label[inner_index] = "B-PHONE_NUM"
                        else:
                            sub.label[inner_index] = "I-PHONE_NUM"

                        if DEBUG:
                            if old_label != sub.label[inner_index]:
                                print(
                                    string_to_check, old_label, sub.label[inner_index]
                                )

                    continue
            except:
                pass

    sub["row_id"] = sub.index
    return sub


# ================================================================================================
def postprocess_street_address(df):

    print("postprocessing street address")

    sub = df
    new_street_addresses = []

    for i in range(len(sub)):
        if sub.label[i] == "B-STREET_ADDRESS":
            start = i
            end = i + 1
            while (
                end < len(sub)
                and sub.label[end] == "I-STREET_ADDRESS"
                and sub.document[end] == sub.document[start]
                and sub.token[end] - sub.token[start] <= 12
            ):
                end += 1
            end -= 1

            token_diff = sub.token[end] - sub.token[start]
            index_diff = end - start
            if 0 <= token_diff - index_diff <= 2:
                for new_index in range(sub.token[start], sub.token[end] + 1):
                    if new_index == sub.token[start]:
                        new_street_addresses.append(
                            [
                                sub.document[start],
                                new_index,
                                "B-STREET_ADDRESS",
                                "\n",
                                0,
                            ]
                        )
                    else:
                        new_street_addresses.append(
                            [
                                sub.document[start],
                                new_index,
                                "I-STREET_ADDRESS",
                                "\n",
                                0,
                            ]
                        )

    sub = pd.concat(
        [
            sub,
            pd.DataFrame(
                new_street_addresses,
                columns=["document", "token", "label", "token_str", "row_id"],
            ),
        ]
    ).reset_index(drop=True)
    sub["row_id"] = sub.index
    return sub


def remove_false_positives(df):

    print("removing long ids (>25) and short urls (<10)")

    sub = df
    sub["valid"] = True
    for i in range(len(sub)):
        if sub.label[i] == "B-ID_NUM" and len(sub.token_str[i]) > 25:
            sub.valid[i] = False

        if sub.label[i] == "B-URL_PERSONAL" and len(sub.token_str[i]) < 10:
            sub.valid[i] = False

    sub = sub[sub.valid == True].reset_index(drop=True)
    sub["row_id"] = sub.index
    return sub


def postprocess_username(df):

    print("postprocessing username")

    sub = df
    for i in range(len(sub)):
        if (
            i + 2 < len(sub)
            and sub.label[i] == "B-USERNAME"
            and sub.label[i + 1] == "B-USERNAME"
            and sub.label[i + 2] == "B-USERNAME"
            and sub.document[i] == sub.document[i + 1]
            and sub.document[i] == sub.document[i + 2]
            and sub.token[i + 1] == sub.token[i] + 1
            and sub.token[i + 2] == sub.token[i] + 2
            and sub.token_str[i + 1] in ["-", "."]
        ):
            sub.label[i + 1] = "I-USERNAME"
            sub.label[i + 2] = "I-USERNAME"

    sub["row_id"] = sub.index

    return sub


def postprocess_same_name(df, doc2tokens, doc_id):

    same_names = []
    doc_piis = set(df["token_str"].unique())
    for i in range(len(doc2tokens[str(doc_id)])):
        cur = doc2tokens[str(doc_id)][i]
        if len(cur) > 1 and cur in doc_piis:
            if len(same_names) > 0 and same_names[-1][1] == i - 1:
                same_names.append([doc_id, i, "I-NAME_STUDENT", cur, 0])
            else:
                same_names.append([doc_id, i, "B-NAME_STUDENT", cur, 0])

    return pd.DataFrame(
        same_names, columns=["document", "token", "label", "token_str", "row_id"]
    )


def postprocess_id_span(df):

    print("postprocessing id span")

    sub = df
    for i in range(len(sub) - 1, 1, -1):
        if (
            sub.document[i] == sub.document[i - 1]
            and sub.token[i] == sub.token[i - 1] + 1
            and sub.label[i] == "B-ID_NUM"
            and sub.label[i - 1] == "B-ID_NUM"
        ):
            sub.label[i] = "I-ID_NUM"
    sub["row_id"] = sub.index
    return sub


def all_postprocess(df):

    df = postprocess_id_phone(df)  # a bit risky
    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = postprocess_street_address(df)  # ok
    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = postprocess_username(df)  # ok
    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = remove_false_positives(df)  # a bit risky
    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = postprocess_id_span(df)  # a bit risky
    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ================================================================================================
def label_postprocessing(df, doc2tokens, data):

    df["keep"] = df.apply(filter_student_preds, axis=1)  # necessary
    df["keep2"] = df.apply(is_valid_id_num, axis=1)  # very risky
    df = (
        df[(df.keep == True) & (df.keep2 == True)]
        .reset_index(drop=True)
        .drop(columns=["keep", "keep2"])
    )

    df = df.drop_duplicates(subset=["document", "token"], keep="first")
    df = df[
        ~((df["label"].str.contains("EMAIL")) & (~df["token_str"].str.contains("@")))
    ]  # ok
    df = df[
        ~(
            (df["label"].str.contains("NAME_STUDENT"))
            & (df["token_str"].isin(subtitles))
        )
    ]  # necessary

    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = all_postprocess(df).drop(columns=["valid"])

    regex_df = regex_predictions(data)

    extras = []
    for row in data:
        sdf = (
            df[
                (df.document == row["document"])
                & ((df.label == "B-NAME_STUDENT") | (df.label == "I-NAME_STUDENT"))
            ]
            .copy()
            .reset_index(drop=True)
        )  # ?
        extras.append(postprocess_same_name(sdf, doc2tokens, row["document"]))

    extras.append(df)
    extras.append(regex_df)

    df = pd.concat(extras).drop_duplicates(subset=["document", "token"], keep="first")
    df.sort_values(by=["document", "token"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["row_id"] = list(range(len(df)))

    return df
