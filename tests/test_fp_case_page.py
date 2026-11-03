from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore, add_item


def _page():
    from fp_case_page import render_case_workspace
    render_case_workspace()


def _widget(items, label):
    return next(w for w in items if w.label == label)


def test_ui_intake_region_and_review_persist(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    monkeypatch.setenv("LV_CASES_DIR", str(root))
    image = tmp_path / "report.png"
    Image.new("RGB", (50, 50), "red").save(image)
    at = AppTest.from_function(_page, default_timeout=30).run()
    assert not at.exception
    at.text_input(key="fp_new_path").set_value(str(image))
    _widget(at.text_input, "案件名稱（可留空）").set_value("UI 誤逮報案")
    _widget(at.button, "建立案件").click().run()
    assert not at.exception and not at.error
    assert len(CaseStore(root).list()) == 1
    _widget(at.text_input, "模型預測類別／爭議名稱").set_value("crack")
    _widget(at.button, "加入爭議區域").click().run()
    assert not at.exception and not at.error
    _widget(at.selectbox, "覆核結果").select("normal")
    _widget(at.button, "儲存此區域覆核").click().run()
    assert at.error and "覆核者" in at.error[0].value
    _widget(at.text_input, "覆核者").set_value("UI Engineer")
    _widget(at.text_area, "判斷依據").set_value("依規格是正常反光")
    _widget(at.button, "儲存此區域覆核").click().run()
    assert not at.exception and not at.error
    case = CaseStore(root).list()[0]
    assert case["items"][0]["review"]["verdict"] == "normal"
    assert case["items"][0]["review"]["reviewer"] == "UI Engineer"
    reopened = AppTest.from_function(_page, default_timeout=30).run()
    assert not reopened.exception
    assert _widget(reopened.selectbox, "覆核結果").value == "normal"


def test_page_entry_never_loads_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    import fp_case_page
    def unexpected(*args, **kwargs):
        raise AssertionError("page entry must not search")
    monkeypatch.setattr(fp_case_page, "search", unexpected)
    at = AppTest.from_function(_page, default_timeout=30).run()
    assert not at.exception
    assert not list((tmp_path / "cases").rglob("*.npy"))


def test_new_case_replaces_previous_case_selection_and_url(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    monkeypatch.setenv("LV_CASES_DIR", str(root))
    image = tmp_path / "report.png"
    Image.new("RGB", (50, 50), "red").save(image)
    store = CaseStore(root)
    original = store.create(store.ingest(image.read_bytes(), image.name), "Existing case")
    at = AppTest.from_function(_page, default_timeout=30)
    at.query_params["inv_case"] = original["id"]
    at.run()
    assert _widget(at.selectbox, "開啟案件").value == original["id"]
    at.text_input(key="fp_new_path").set_value(str(image))
    _widget(at.text_input, "案件名稱（可留空）").set_value("New case")
    _widget(at.button, "建立案件").click().run()
    assert not at.exception and not at.error
    created = next(c for c in store.list() if c["title"] == "New case")
    assert _widget(at.selectbox, "開啟案件").value == created["id"]
    assert at.query_params["inv_case"] == [created["id"]]
