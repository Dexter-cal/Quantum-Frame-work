"""Tests for qai.mcp — Model Context Protocol server + tools."""
import sys, os, unittest, tempfile, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.mcp import (
    MCPServer, Tool, ToolRegistry, tool_registry, create_server,
    JsonRpcRequest, JsonRpcResponse, JsonRpcError,
    FilesystemTool, MemoryTool, GitTool, FetchTool, TimeTool,
    SqliteTool, ExecTool, SequentialThinkingTool,
    CalculatorTool, CodeEvalTool,
    QaiModelTool, QaiLogicTool, QaiMemoryTool, QaiBundleTool,
)


class TestJsonRpcProtocol(unittest.TestCase):
    def test_request_from_dict(self):
        req = JsonRpcRequest.from_dict({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1,
        })
        self.assertEqual(req.method, "tools/list")
        self.assertEqual(req.id, 1)

    def test_response_to_dict(self):
        resp = JsonRpcResponse(result={"x": 1}, id=1)
        d = resp.to_dict()
        self.assertEqual(d["jsonrpc"], "2.0")
        self.assertEqual(d["id"], 1)
        self.assertEqual(d["result"], {"x": 1})
        self.assertNotIn("error", d)

    def test_error_response(self):
        resp = JsonRpcResponse(error={"code": -32601, "message": "not found"}, id=1)
        d = resp.to_dict()
        self.assertIn("error", d)
        self.assertNotIn("result", d)

    def test_error_to_dict(self):
        err = JsonRpcError(code=-32601, message="method not found", data="extra")
        d = err.to_dict()
        self.assertEqual(d["code"], -32601)
        self.assertEqual(d["message"], "method not found")
        self.assertEqual(d["data"], "extra")


class TestToolRegistry(unittest.TestCase):
    def test_register_and_create(self):
        class MyTool(Tool):
            name = "my_test_tool"
            description = "test"
            def run(self, **kw):
                return "ok"
        reg = ToolRegistry()
        reg.register_class("my_test_tool", MyTool)
        tool = reg.create("my_test_tool")
        self.assertIsInstance(tool, MyTool)

    def test_unknown_tool_raises(self):
        reg = ToolRegistry()
        with self.assertRaises(ValueError):
            reg.create("nonexistent")

    def test_module_registry_has_builtins(self):
        names = tool_registry.list_classes()
        for n in ["filesystem", "memory", "git", "fetch", "time",
                  "sqlite", "exec", "sequential_thinking",
                  "calculator", "code_eval",
                  "qai_model", "qai_logic", "qai_memory", "qai_bundle"]:
            self.assertIn(n, names, f"missing built-in tool: {n}")


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        result = self.tool.run(expression="2 + 3 * 4")
        self.assertEqual(result["value"], 14)

    def test_float_division(self):
        result = self.tool.run(expression="10 / 3")
        self.assertAlmostEqual(result["value"], 3.333333, places=4)

    def test_powers_and_parens(self):
        result = self.tool.run(expression="(2 + 3) ** 2")
        self.assertEqual(result["value"], 25)

    def test_math_functions(self):
        result = self.tool.run(expression="sqrt(16) + abs(-3)")
        self.assertEqual(result["value"], 7)

    def test_constants(self):
        result = self.tool.run(expression="pi")
        self.assertAlmostEqual(result["value"], 3.14159265, places=4)

    def test_rejects_import(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(expression="__import__('os').system('echo hack')")

    def test_rejects_attribute_access(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(expression="(1).__class__.__bases__")

    def test_syntax_error(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(expression="2 +")


class TestTimeTool(unittest.TestCase):
    def setUp(self):
        self.tool = TimeTool()

    def test_now(self):
        result = self.tool.run(operation="now")
        self.assertIn("iso", result)
        self.assertIn("unix", result)
        self.assertGreater(result["unix"], 0)

    def test_now_with_timezone(self):
        result = self.tool.run(operation="now", timezone="UTC")
        self.assertIn("iso", result)

    def test_format(self):
        result = self.tool.run(operation="format", timestamp=0, fmt="%Y-%m-%d")
        self.assertEqual(result, "1970-01-01")

    def test_invalid_operation(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(operation="invalid")


class TestFilesystemTool(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = FilesystemTool(allowed_dirs=[self.tmpdir])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_read(self):
        path = os.path.join(self.tmpdir, "test.txt")
        self.tool.run(operation="write", path=path, content="hello")
        result = self.tool.run(operation="read", path=path)
        self.assertEqual(result, "hello")

    def test_list(self):
        self.tool.run(operation="write", path=os.path.join(self.tmpdir, "a.txt"), content="a")
        self.tool.run(operation="write", path=os.path.join(self.tmpdir, "b.txt"), content="b")
        result = self.tool.run(operation="list", path=self.tmpdir)
        names = [e["name"] for e in result]
        self.assertIn("a.txt", names)
        self.assertIn("b.txt", names)

    def test_exists(self):
        path = os.path.join(self.tmpdir, "test.txt")
        self.tool.run(operation="write", path=path, content="x")
        result = self.tool.run(operation="exists", path=path)
        self.assertTrue(result["exists"])

    def test_mkdir_delete(self):
        path = os.path.join(self.tmpdir, "subdir")
        self.tool.run(operation="mkdir", path=path)
        self.assertTrue(os.path.isdir(path))
        self.tool.run(operation="delete", path=path)
        self.assertFalse(os.path.exists(path))

    def test_path_outside_sandbox_rejected(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(operation="read", path="/etc/passwd")


class TestMemoryTool(unittest.TestCase):
    def setUp(self):
        self.tool = MemoryTool()

    def test_create_entity(self):
        result = self.tool.run(operation="create_entity", entity_name="Alice",
                              entity_type="person", observations=["is a researcher"])
        self.assertEqual(result["created"], "Alice")

    def test_create_relation(self):
        self.tool.run(operation="create_entity", entity_name="Alice", entity_type="person")
        self.tool.run(operation="create_entity", entity_name="Bob", entity_type="person")
        result = self.tool.run(operation="create_relation", subject="Alice",
                              predicate="knows", object="Bob")
        self.assertEqual(result["created"]["predicate"], "knows")

    def test_get_entity_includes_relations(self):
        self.tool.run(operation="create_entity", entity_name="Alice", entity_type="person")
        self.tool.run(operation="create_entity", entity_name="Bob", entity_type="person")
        self.tool.run(operation="create_relation", subject="Alice", predicate="knows", object="Bob")
        result = self.tool.run(operation="get_entity", entity_name="Alice")
        self.assertEqual(len(result["relations"]), 1)

    def test_search(self):
        self.tool.run(operation="create_entity", entity_name="Paris", entity_type="city",
                      observations=["capital of France"])
        result = self.tool.run(operation="search", query="France")
        self.assertEqual(len(result), 1)

    def test_persist_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tool = MemoryTool(persist_path=path)
            tool.run(operation="create_entity", entity_name="X", entity_type="thing")
            self.assertTrue(os.path.exists(path))
            # Reload
            tool2 = MemoryTool(persist_path=path)
            self.assertIn("X", tool2.entities)
        finally:
            os.unlink(path)


class TestSequentialThinkingTool(unittest.TestCase):
    def setUp(self):
        self.tool = SequentialThinkingTool()

    def test_think(self):
        result = self.tool.run(operation="think", thought="hello",
                              thought_number=1, next_thought_needed=False)
        self.assertEqual(result["thought"], "hello")
        self.assertEqual(result["thought_number"], 1)

    def test_list(self):
        self.tool.run(operation="think", thought="A", thought_number=1)
        self.tool.run(operation="think", thought="B", thought_number=2)
        result = self.tool.run(operation="list")
        self.assertEqual(len(result), 2)

    def test_summary(self):
        self.tool.run(operation="think", thought="A", thought_number=1, next_thought_needed=True)
        self.tool.run(operation="think", thought="B", thought_number=2, next_thought_needed=False)
        result = self.tool.run(operation="summary")
        self.assertEqual(result["total_thoughts"], 2)
        self.assertTrue(result["completed"])

    def test_clear(self):
        self.tool.run(operation="think", thought="A", thought_number=1)
        result = self.tool.run(operation="clear")
        self.assertEqual(result["cleared"], 1)


class TestCodeEvalTool(unittest.TestCase):
    def setUp(self):
        self.tool = CodeEvalTool()

    def test_basic_arithmetic(self):
        result = self.tool.run(code="x = 5\ny = 10\nresult = x + y")
        self.assertEqual(result.get("result"), 15)

    def test_print_captured(self):
        result = self.tool.run(code='print("hello world")')
        self.assertIn("hello world", result["stdout"])

    def test_forbidden_import(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(code="import os\nos.system('echo hack')")

    def test_forbidden_eval(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(code="eval('1+1')")


class TestExecTool(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = ExecTool(allowed_work_dirs=[self.tmpdir])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_echo(self):
        result = self.tool.run(command="echo hello", work_dir=self.tmpdir)
        self.assertIn("hello", result["stdout"])
        self.assertEqual(result["exit_code"], 0)

    def test_work_dir_restriction(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(command="ls", work_dir="/etc")


class TestFetchTool(unittest.TestCase):
    def setUp(self):
        self.tool = FetchTool()

    def test_invalid_url(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(url="not-a-url")

    def test_invalid_scheme(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(url="ftp://example.com")


class TestSqliteTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        import sqlite3
        conn = sqlite3.connect(self.tmp.name)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users (name) VALUES ('Alice'), ('Bob')")
        conn.commit()
        conn.close()
        self.tool = SqliteTool()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_tables(self):
        result = self.tool.run(operation="tables", db_path=self.tmp.name)
        self.assertIn("users", result)

    def test_query(self):
        result = self.tool.run(operation="query", db_path=self.tmp.name,
                              sql="SELECT * FROM users")
        self.assertEqual(result["rowcount"], 2)

    def test_rejects_write(self):
        with self.assertRaises(JsonRpcError):
            self.tool.run(operation="query", db_path=self.tmp.name,
                         sql="DELETE FROM users")


class TestQaiModelTool(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)
        self.tool = QaiModelTool(model=self.model, tool_name="predict")

    def test_predict_single(self):
        result = self.tool.run(X=[4.0])
        self.assertAlmostEqual(result, 8.0, places=4)

    def test_predict_batch(self):
        result = self.tool.run(X=[[1.0], [2.0]])
        self.assertEqual(len(result), 2)


class TestQaiLogicTool(unittest.TestCase):
    def setUp(self):
        @qai.logic
        def my_logic(x):
            """Double a number."""
            return float(x) * 2
        self.tool = QaiLogicTool(logic_fn=my_logic)

    def test_call(self):
        result = self.tool.run(x="21")
        self.assertEqual(result, 42.0)


class TestQaiMemoryTool(unittest.TestCase):
    def setUp(self):
        self.tool = QaiMemoryTool()

    def test_working_add_recent(self):
        self.tool.run(operation="working_add", text="hello")
        self.tool.run(operation="working_add", text="world")
        result = self.tool.run(operation="working_recent", n=10)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_retrieval_add_search(self):
        self.tool.run(operation="retrieval_add", text="Paris is the capital of France")
        result = self.tool.run(operation="retrieval_search", query="capital", k=1)
        self.assertEqual(len(result), 1)
        self.assertIn("Paris", result[0]["text"])


class TestQaiBundleTool(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False)
        self.tmp.close()
        qai.save_bundle(self.model, self.tmp.name, sign_key="secret")
        self.tool = QaiBundleTool()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_inspect(self):
        result = self.tool.run(operation="inspect", path=self.tmp.name)
        self.assertEqual(result["manifest"]["technique"], "regression")

    def test_predict(self):
        result = self.tool.run(operation="predict", path=self.tmp.name,
                              X=[4.0], sign_key="secret")
        self.assertAlmostEqual(result, 8.0, places=4)


class TestMCPServer(unittest.TestCase):
    def setUp(self):
        self.server = create_server(
            name="test-server",
            tools=[CalculatorTool(), TimeTool()],
        )

    def test_register_unregister(self):
        n_before = len(self.server._tools)
        self.server.register(SequentialThinkingTool())
        self.assertEqual(len(self.server._tools), n_before + 1)
        self.server.unregister("sequential_thinking")
        self.assertEqual(len(self.server._tools), n_before)

    def test_list_tools(self):
        tools = self.server.list_tools()
        names = [t["name"] for t in tools]
        self.assertIn("calculator", names)
        self.assertIn("time", names)

    def test_call_tool(self):
        result = self.server.call_tool("calculator", {"expression": "2+2"})
        self.assertEqual(result["value"], 4)

    def test_call_unknown_tool(self):
        with self.assertRaises(JsonRpcError):
            self.server.call_tool("nonexistent", {})

    def test_inspect_tool(self):
        info = self.server.inspect_tool("calculator")
        self.assertEqual(info["name"], "calculator")
        self.assertIn("inputSchema", info)

    def test_handle_raw_initialize(self):
        resp = self.server.handle_raw({"jsonrpc": "2.0", "method": "initialize", "id": 0})
        self.assertEqual(resp["result"]["serverInfo"]["name"], "test-server")

    def test_handle_raw_tools_list(self):
        resp = self.server.handle_raw({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        self.assertIn("tools", resp["result"])

    def test_handle_raw_tools_call(self):
        resp = self.server.handle_raw({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "5*5"}},
            "id": 2,
        })
        self.assertIn("result", resp)

    def test_handle_raw_unknown_method(self):
        resp = self.server.handle_raw({"jsonrpc": "2.0", "method": "unknown", "id": 3})
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)


class TestPermissionsAndRateLimit(unittest.TestCase):
    def test_denied_tools(self):
        server = create_server(
            tools=[CalculatorTool(), TimeTool()],
            denied_tools=["time"],
        )
        with self.assertRaises(JsonRpcError) as ctx:
            server.call_tool("time", {"operation": "now"})
        self.assertEqual(ctx.exception.code, -32003)  # PERMISSION_DENIED

    def test_allowed_tools(self):
        server = create_server(
            tools=[CalculatorTool(), TimeTool()],
            allowed_tools=["calculator"],
        )
        # calculator works
        server.call_tool("calculator", {"expression": "1+1"})
        # time denied
        with self.assertRaises(JsonRpcError):
            server.call_tool("time", {"operation": "now"})

    def test_rate_limit(self):
        server = create_server(tools=[CalculatorTool()], default_rate_limit=3)
        server._tools["calculator"].rate_limit_per_minute = 3
        for i in range(3):
            server.call_tool("calculator", {"expression": str(i)})
        with self.assertRaises(JsonRpcError) as ctx:
            server.call_tool("calculator", {"expression": "999"})
        self.assertEqual(ctx.exception.code, -32004)  # RATE_LIMIT_EXCEEDED


class TestAuditLog(unittest.TestCase):
    def test_audit_log_records_calls(self):
        server = create_server(tools=[CalculatorTool()])
        server.call_tool("calculator", {"expression": "1+1"})
        log = server.get_audit_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["tool"], "calculator")
        self.assertIsNone(log[0]["error"])

    def test_audit_log_records_errors(self):
        server = create_server(tools=[CalculatorTool()])
        try:
            server.call_tool("calculator", {"expression": "import os"})
        except JsonRpcError:
            pass
        log = server.get_audit_log()
        self.assertEqual(len(log), 1)
        self.assertIsNotNone(log[0]["error"])

    def test_clear_audit_log(self):
        server = create_server(tools=[CalculatorTool()])
        server.call_tool("calculator", {"expression": "1+1"})
        n = server.clear_audit_log()
        self.assertEqual(n, 1)
        self.assertEqual(len(server.get_audit_log()), 0)

    def test_audit_log_size_limit(self):
        server = create_server(tools=[CalculatorTool()], audit_log_size=5)
        for i in range(10):
            server.call_tool("calculator", {"expression": str(i)})
        log = server.get_audit_log()
        self.assertLessEqual(len(log), 5)


class TestHttpTransport(unittest.TestCase):
    def setUp(self):
        self.server = create_server(tools=[CalculatorTool()])

    def test_build_http_app(self):
        app = self.server.build_http_app()
        self.assertIsNotNone(app)

    def test_health_endpoint(self):
        app = self.server.build_http_app()
        client = app.test_client()
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_tools_endpoint(self):
        app = self.server.build_http_app()
        client = app.test_client()
        resp = client.get("/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(len(data["tools"]), 0)

    def test_rpc_endpoint(self):
        app = self.server.build_http_app()
        client = app.test_client()
        resp = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "2+2"}},
            "id": 1,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("result", data)


class TestIntegrationWithQai(unittest.TestCase):
    def test_accessible_from_top_level(self):
        self.assertTrue(hasattr(qai, "mcp"))
        self.assertTrue(hasattr(qai, "MCPServer"))
        self.assertTrue(hasattr(qai, "CalculatorTool"))
        self.assertTrue(hasattr(qai, "QaiModelTool"))

    def test_register_qai_model_helper(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        server = qai.mcp.create_server(name="test")
        server.register_qai_model(m, tool_name="predict")
        result = server.call_tool("predict", {"X": [3.0]})
        self.assertAlmostEqual(result, 6.0, places=4)

    def test_register_qai_logic_helper(self):
        @qai.logic
        def double(x):
            return float(x) * 2
        server = qai.mcp.create_server(name="test")
        server.register_qai_logic(double, tool_name="double")
        result = server.call_tool("double", {"x": "5"})
        self.assertEqual(result, 10.0)


if __name__ == "__main__":
    unittest.main()
