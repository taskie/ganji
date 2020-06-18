"""Load TTF as ndarray."""

from typing import Dict, List, Tuple

import freetype
import numpy as np


def _calc_copy_nd(
    src: Tuple[int, ...], dst: Tuple[int, ...]
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    if len(src) != len(dst):
        raise Exception("invalid dimension")
    src_result: List[Tuple[int, int]] = []
    dst_result: List[Tuple[int, int]] = []
    for i in range(len(src)):
        src_x = src[i]
        dst_x = dst[i]
        if dst_x > src_x:
            begin = (dst_x - src_x) // 2
            end = begin + src_x
            src_result.append((0, src_x))
            dst_result.append((begin, end))
        else:
            begin = (src_x - dst_x) // 2
            end = begin + dst_x
            src_result.append((begin, end))
            dst_result.append((0, dst_x))
    return (tuple(src_result), tuple(dst_result))


def _copy_2d(src: np.ndarray, dst: np.ndarray):
    s, d = _calc_copy_nd(src.shape, dst.shape)
    dst[d[0][0] : d[0][1], d[1][0] : d[1][1]] = src[s[0][0] : s[0][1], s[1][0] : s[1][1]]


def _setup_face(font_path: str, size: int) -> freetype.Face:
    face = freetype.Face(font_path)
    face.set_char_size(size * 64)
    return face


def _make_glyph_bitmap(face: freetype.Face, codepoint: int) -> np.ndarray:
    char_index = face.get_char_index(codepoint)
    if char_index == 0:
        return None
    face.load_glyph(char_index)
    bitmap = face.glyph.bitmap
    length = len(bitmap.buffer)
    if length == 0:
        return None
    row_count = bitmap.rows
    col_count = length // row_count
    glyph_size = (row_count, col_count)
    glyph_bitmap = np.reshape(np.array(bitmap.buffer, dtype=np.uint8), glyph_size)
    return glyph_bitmap


def _convert_codepoints_from_str(s: str) -> List[int]:
    return [ord(c) for c in s if not c.isspace()]


def _convert_codepoints_from_ranges(codepoint_ranges: List[Tuple[int, int]]) -> List[int]:
    return [
        codepoint for codepoint_range in codepoint_ranges for codepoint in range(codepoint_range[0], codepoint_range[1])
    ]


def _make_glyph_bitmap_dict(face: freetype.Face, codepoints: List[int]) -> Dict[int, np.ndarray]:
    result = {}
    for codepoint in codepoints:
        try:
            glyph_bitmap = _make_glyph_bitmap(face, codepoint)
            if glyph_bitmap is None:
                continue
            result[codepoint] = glyph_bitmap
        except freetype.FT_Exception:
            pass
    return result


def _convert_bitmap_value_to_str(x: int) -> str:
    if x > 192:
        return "**"
    elif x > 64:
        return "++"
    elif x > 0:
        return "--"
    else:
        return "  "


def _convert_bitmap_to_asciiart(bitmap: np.ndarray) -> str:
    s = ""
    for i in range(bitmap.shape[0]):
        s += "".join(_convert_bitmap_value_to_str(x) for x in bitmap[i]) + "\n"
    return s


def load_data_for_gan(
    font_path: str, size: int, codepoints: List[int], *, thickness_min: float = None, thickness_max: float = None
) -> np.ndarray:
    face = _setup_face(font_path, size)
    glyph_bitmap_dict = _make_glyph_bitmap_dict(face, codepoints)
    items = []
    if thickness_min is None and thickness_max is None:
        for (_codepoint, glyph_bitmap) in sorted(glyph_bitmap_dict.items()):
            items.append(glyph_bitmap)
    else:
        for (_codepoint, glyph_bitmap) in sorted(glyph_bitmap_dict.items()):
            thickness = np.average(glyph_bitmap)
            if thickness_min <= thickness < thickness_max:
                items.append(glyph_bitmap)
    count = len(items)
    data = np.zeros((count, size, size, 1), dtype=np.uint8)
    for i, glyph_bitmap in enumerate(items):
        _copy_2d(glyph_bitmap, data[i, :, :, 0])
    return data


def load_data_for_pix2pix(font_paths: List[str], size: int, codepoints: List[int]) -> np.ndarray:
    glyph_bitmap_dicts: List[Dict[int, np.ndarray]] = []
    for font_path in font_paths:
        face = _setup_face(font_path, size)
        glyph_bitmap_dict = _make_glyph_bitmap_dict(face, codepoints)
        glyph_bitmap_dicts.append(glyph_bitmap_dict)
    common_codepoints = set(glyph_bitmap_dicts[0].keys())
    for i in range(1, len(glyph_bitmap_dicts)):
        common_codepoints.intersection_update(glyph_bitmap_dicts[i].keys())
    count = len(common_codepoints)
    data = np.zeros((len(font_paths), count, size, size, 1), dtype=np.uint8)
    for dict_index, glyph_bitmap_dict in enumerate(glyph_bitmap_dicts):
        for i, codepoint in enumerate(sorted(common_codepoints)):
            glyph_bitmap = glyph_bitmap_dict[codepoint]
            _copy_2d(glyph_bitmap, data[dict_index, i, :, :, 0])
    return data


KANJI_CODEPOINT_RANGES = [
    (0x3400, 0x4DC0),
    (0x4E00, 0xA000),
    (0xF900, 0xFB00),
    (0x20000, 0x30000),
]
KANJI_CODEPOINTS = _convert_codepoints_from_ranges(KANJI_CODEPOINT_RANGES)

JOUYOU_KANJI_STR = """
亜哀挨愛曖悪握圧扱宛嵐安案暗以衣位囲医依委威為畏胃尉異移萎偉椅彙意違維慰遺緯域育一壱逸茨芋引印因咽姻員院淫陰飲隠韻右宇
羽雨唄鬱畝浦運雲永泳英映栄営詠影鋭衛易疫益液駅悦越謁閲円延沿炎宴怨媛援園煙猿遠鉛塩演縁艶汚王凹央応往押旺欧殴桜翁奥横岡
屋億憶臆虞乙俺卸音恩温穏下化火加可仮何花佳価果河苛科架夏家荷華菓貨渦過嫁暇禍靴寡歌箇稼課蚊牙瓦我画芽賀雅餓介回灰会快戒
改怪拐悔海界皆械絵開階塊楷解潰壊懐諧貝外劾害崖涯街慨蓋該概骸垣柿各角拡革格核殻郭覚較隔閣確獲嚇穫学岳楽額顎掛潟括活喝渇
割葛滑褐轄且株釜鎌刈干刊甘汗缶完肝官冠巻看陥乾勘患貫寒喚堪換敢棺款間閑勧寛幹感漢慣管関歓監緩憾還館環簡観韓艦鑑丸含岸岩
玩眼頑顔願企伎危机気岐希忌汽奇祈季紀軌既記起飢鬼帰基寄規亀喜幾揮期棋貴棄毀旗器畿輝機騎技宜偽欺義疑儀戯擬犠議菊吉喫詰却
客脚逆虐九久及弓丘旧休吸朽臼求究泣急級糾宮救球給嗅窮牛去巨居拒拠挙虚許距魚御漁凶共叫狂京享供協況峡挟狭恐恭胸脅強教郷境
橋矯鏡競響驚仰暁業凝曲局極玉巾斤均近金菌勤琴筋僅禁緊錦謹襟吟銀区句苦駆具惧愚空偶遇隅串屈掘窟熊繰君訓勲薫軍郡群兄刑形系
径茎係型契計恵啓掲渓経蛍敬景軽傾携継詣慶憬稽憩警鶏芸迎鯨隙劇撃激桁欠穴血決結傑潔月犬件見券肩建研県倹兼剣拳軒健険圏堅検
嫌献絹遣権憲賢謙鍵繭顕験懸元幻玄言弦限原現舷減源厳己戸古呼固孤弧股虎故枯個庫湖雇誇鼓錮顧五互午呉後娯悟碁語誤護口工公勾
孔功巧広甲交光向后好江考行坑孝抗攻更効幸拘肯侯厚恒洪皇紅荒郊香候校耕航貢降高康控梗黄喉慌港硬絞項溝鉱構綱酵稿興衡鋼講購
乞号合拷剛傲豪克告谷刻国黒穀酷獄骨駒込頃今困昆恨根婚混痕紺魂墾懇左佐沙査砂唆差詐鎖座挫才再災妻采砕宰栽彩採済祭斎細菜最
裁債催塞歳載際埼在材剤財罪崎作削昨柵索策酢搾錯咲冊札刷刹拶殺察撮擦雑皿三山参桟蚕惨産傘散算酸賛残斬暫士子支止氏仕史司四
市矢旨死糸至伺志私使刺始姉枝祉肢姿思指施師恣紙脂視紫詞歯嗣試詩資飼誌雌摯賜諮示字寺次耳自似児事侍治持時滋慈辞磁餌璽鹿式
識軸七叱失室疾執湿嫉漆質実芝写社車舎者射捨赦斜煮遮謝邪蛇尺借酌釈爵若弱寂手主守朱取狩首殊珠酒腫種趣寿受呪授需儒樹収囚州
舟秀周宗拾秋臭修袖終羞習週就衆集愁酬醜蹴襲十汁充住柔重従渋銃獣縦叔祝宿淑粛縮塾熟出述術俊春瞬旬巡盾准殉純循順準潤遵処初
所書庶暑署緒諸女如助序叙徐除小升少召匠床抄肖尚招承昇松沼昭宵将消症祥称笑唱商渉章紹訟勝掌晶焼焦硝粧詔証象傷奨照詳彰障憧
衝賞償礁鐘上丈冗条状乗城浄剰常情場畳蒸縄壌嬢錠譲醸色拭食植殖飾触嘱織職辱尻心申伸臣芯身辛侵信津神唇娠振浸真針深紳進森診
寝慎新審震薪親人刃仁尽迅甚陣尋腎須図水吹垂炊帥粋衰推酔遂睡穂随髄枢崇数据杉裾寸瀬是井世正生成西声制姓征性青斉政星牲省凄
逝清盛婿晴勢聖誠精製誓静請整醒税夕斥石赤昔析席脊隻惜戚責跡積績籍切折拙窃接設雪摂節説舌絶千川仙占先宣専泉浅洗染扇栓旋船
戦煎羨腺詮践箋銭潜線遷選薦繊鮮全前善然禅漸膳繕狙阻祖租素措粗組疎訴塑遡礎双壮早争走奏相荘草送倉捜挿桑巣掃曹曽爽窓創喪痩
葬装僧想層総遭槽踪操燥霜騒藻造像増憎蔵贈臓即束足促則息捉速側測俗族属賊続卒率存村孫尊損遜他多汰打妥唾堕惰駄太対体耐待怠
胎退帯泰堆袋逮替貸隊滞態戴大代台第題滝宅択沢卓拓託濯諾濁但達脱奪棚誰丹旦担単炭胆探淡短嘆端綻誕鍛団男段断弾暖談壇地池知
値恥致遅痴稚置緻竹畜逐蓄築秩窒茶着嫡中仲虫沖宙忠抽注昼柱衷酎鋳駐著貯丁弔庁兆町長挑帳張彫眺釣頂鳥朝貼超腸跳徴嘲潮澄調聴
懲直勅捗沈珍朕陳賃鎮追椎墜通痛塚漬坪爪鶴低呈廷弟定底抵邸亭貞帝訂庭逓停偵堤提程艇締諦泥的笛摘滴適敵溺迭哲鉄徹撤天典店点
展添転塡田伝殿電斗吐妬徒途都渡塗賭土奴努度怒刀冬灯当投豆東到逃倒凍唐島桃討透党悼盗陶塔搭棟湯痘登答等筒統稲踏糖頭謄藤闘
騰同洞胴動堂童道働銅導瞳峠匿特得督徳篤毒独読栃凸突届屯豚頓貪鈍曇丼那奈内梨謎鍋南軟難二尼弐匂肉虹日入乳尿任妊忍認寧熱年
念捻粘燃悩納能脳農濃把波派破覇馬婆罵拝杯背肺俳配排敗廃輩売倍梅培陪媒買賠白伯拍泊迫剝舶博薄麦漠縛爆箱箸畑肌八鉢発髪伐抜
罰閥反半氾犯帆汎伴判坂阪板版班畔般販斑飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼披肥非卑飛疲秘被悲扉費碑罷避尾眉美備微鼻膝肘匹
必泌筆姫百氷表俵票評漂標苗秒病描猫品浜貧賓頻敏瓶不夫父付布扶府怖阜附訃負赴浮婦符富普腐敷膚賦譜侮武部舞封風伏服副幅復福
腹複覆払沸仏物粉紛雰噴墳憤奮分文聞丙平兵併並柄陛閉塀幣弊蔽餅米壁璧癖別蔑片辺返変偏遍編弁便勉歩保哺捕補舗母募墓慕暮簿方
包芳邦奉宝抱放法泡胞俸倣峰砲崩訪報蜂豊飽褒縫亡乏忙坊妨忘防房肪某冒剖紡望傍帽棒貿貌暴膨謀頰北木朴牧睦僕墨撲没勃堀本奔翻
凡盆麻摩磨魔毎妹枚昧埋幕膜枕又末抹万満慢漫未味魅岬密蜜脈妙民眠矛務無夢霧娘名命明迷冥盟銘鳴滅免面綿麺茂模毛妄盲耗猛網目
黙門紋問冶夜野弥厄役約訳薬躍闇由油喩愉諭輸癒唯友有勇幽悠郵湧猶裕遊雄誘憂融優与予余誉預幼用羊妖洋要容庸揚揺葉陽溶腰様瘍
踊窯養擁謡曜抑沃浴欲翌翼拉裸羅来雷頼絡落酪辣乱卵覧濫藍欄吏利里理痢裏履璃離陸立律慄略柳流留竜粒隆硫侶旅虜慮了両良料涼猟
陵量僚領寮療瞭糧力緑林厘倫輪隣臨瑠涙累塁類令礼冷励戻例鈴零霊隷齢麗暦歴列劣烈裂恋連廉練錬呂炉賂路露老労弄郎朗浪廊楼漏籠
六録麓論和話賄脇惑枠湾腕
"""
JOUYOU_KANJI_CODEPOINTS = _convert_codepoints_from_str(JOUYOU_KANJI_STR)

HIRAGANA_CODEPOINT_RANGES = [(0x3041, 0x3097)]
HIRAGANA_CODEPOINTS = _convert_codepoints_from_ranges(HIRAGANA_CODEPOINT_RANGES)

CODEPOINTS_MAP = {
    "kanji": KANJI_CODEPOINTS,
    "jouyou_kanji": JOUYOU_KANJI_CODEPOINTS,
    "hiragana": HIRAGANA_CODEPOINTS,
}


def find_codepoints(key: str, *, map: Dict[str, List[int]] = CODEPOINTS_MAP) -> List[int]:
    return CODEPOINTS_MAP[key.replace("-", "_").lower()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FreeType loader")
    parser.add_argument(
        "-c",
        "--codepoint-set",
        help="codepoint set (kanji|jouyou-kanji|hiragana) [default: hiragana]",
        default="hiragana",
    )
    parser.add_argument("-S", "--size", type=int, help="size [default: 32]", default=32)
    parser.add_argument("fonts", help="font files", nargs="*")
    args = parser.parse_args()
    codepoints = find_codepoints(args.codepoint_set)
    for font in args.fonts:
        data = load_data_for_gan(font, args.size, codepoints)
        print(data.shape)
        for i in range(data.shape[0]):
            print(_convert_bitmap_to_asciiart(data[i, :, :, 0]))
