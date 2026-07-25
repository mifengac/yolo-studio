/* 全局壳层：路由、API、状态 */
const YS = {
  api: async function (url, opts = {}) {
    const res = await fetch(url, opts);
    let data = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      data = await res.json();
    } else {
      data = await res.text();
    }
    if (!res.ok) {
      const msg = (data && data.detail) || (typeof data === "string" ? data : JSON.stringify(data));
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  },
  statusCn(s) {
    return ({
      pending: "排队中",
      running: "运行中",
      success: "成功",
      failed: "失败",
      canceled: "已取消",
      interrupted: "已中断",
    })[s] || s;
  },
  fmtSec(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (h) return `${h}小时${m}分`;
    return `${m}分${sec % 60}秒`;
  },
};

window.YSApp = null;

document.addEventListener("DOMContentLoaded", () => {
  const { createApp } = Vue;
  window.YSApp = createApp({
    data() {
      return {
        page: "datasets",
        datasets: [],
        currentDs: null,
        showCreate: false,
        newDs: { name: "", classes: "wheelie, multi rider", notes: "" },
        toast: "",
        taskMsg: {},
        sysInfo: null,
        trainForm: {
          dataset_id: "",
          group: "finetune",
          base_model: "0517_yolo26n_wheelie_multi-rider.pt",
          epochs: 20,
          imgsz: 416,
          batch: 16,
          freeze: 10,
          only_confirmed: true,
          force_long: false,
        },
        baseModels: { scratch: [], finetune: [] },
        estimate: null,
        trainJobs: [],
        activeJob: null,
        trainLog: "",
        modelList: [],
        _logEs: null,
        _chart: null,
        _poll: null,
        _jobsPoll: null,
        evalDatasetId: "",
        cropPanel: {
          show: false,
          dest: null,
          source: "zip",
          srcDatasetId: "",
          zipFile: null,
          busy: false,
          msg: "",
          taskId: null,
          preview: null,
          params: {
            model: "yolo26n.pt",
            target_class: "person",
            conf: 0.25,
            imgsz: 1280,
            min_box_h: 100,
            pad_ratio: 0.25,
            top_ratio: -0.15,
            bottom_ratio: 0.55,
            max_crops: 8000,
            preview_limit: 20,
          },
        },
        alPanel: {
          show: false,
          ds: null,
          model: "default",
          conf: 0.15,
          overwrite: false,
          options: [],
        },
      };
    },
    computed: {
      baseModelOptions() {
        return this.trainForm.group === "scratch"
          ? this.baseModels.scratch || []
          : this.baseModels.finetune || [];
      },
    },
    methods: {
      statusCn: YS.statusCn,
      fmtSec: YS.fmtSec,
      showToast(msg) {
        this.toast = msg;
        setTimeout(() => { if (this.toast === msg) this.toast = ""; }, 3200);
      },
      go(page) {
        // 离开标注页时卸载监听，避免重复绑定
        if (this.page === "labeler" && page !== "labeler" && window.YSLabeler) {
          window.YSLabeler.destroy();
        }
        this.page = page;
        if (this._jobsPoll) {
          clearInterval(this._jobsPoll);
          this._jobsPoll = null;
        }
        if (page === "datasets") this.loadDatasets();
        if (page === "train") {
          this.loadDatasets();
          this.loadBaseModels();
          this.loadJobs();
          if (this.currentDs) this.trainForm.dataset_id = this.currentDs.id;
          this.$nextTick(() => this.refreshEstimate());
          // 训练页自动刷新任务列表进度
          this._jobsPoll = setInterval(() => this.loadJobs(), 5000);
        }
        if (page === "models") this.loadModels();
        if (page === "labeler" && this.currentDs) {
          this.$nextTick(() => {
            if (window.YSLabeler) window.YSLabeler.mount(this.currentDs, this);
          });
        }
      },
      async loadDatasets() {
        this.datasets = await YS.api("/api/datasets");
      },
      async createDataset() {
        try {
          const classes = this.newDs.classes.split(/[,，;；]/).map((s) => s.trim()).filter(Boolean);
          const d = await YS.api("/api/datasets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: this.newDs.name, classes, notes: this.newDs.notes }),
          });
          this.showCreate = false;
          this.newDs = { name: "", classes: "wheelie, multi rider", notes: "" };
          await this.loadDatasets();
          this.showToast("数据集已创建");
          this.openLabeler(d);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      openLabeler(d) {
        this.currentDs = d;
        this.trainForm.dataset_id = d.id;
        this.go("labeler");
      },
      async removeDataset(d) {
        if (!confirm(`确定删除数据集「${d.name}」？图片与标注会一并删除。`)) return;
        await YS.api(`/api/datasets/${d.id}`, { method: "DELETE" });
        await this.loadDatasets();
        if (this.currentDs && this.currentDs.id === d.id) this.currentDs = null;
        this.showToast("已删除");
      },
      async uploadFiles(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "image/*";
        input.onchange = async () => {
          const fd = new FormData();
          for (const f of input.files) fd.append("files", f);
          try {
            const r = await YS.api(`/api/datasets/${d.id}/import/files`, { method: "POST", body: fd });
            this.showToast(`导入 ${r.imported} 张（跳过重复 ${r.skipped_dup}）`);
            await this.loadDatasets();
          } catch (e) {
            this.showToast(e.message);
          }
        };
        input.click();
      },
      async uploadZip(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".zip";
        input.onchange = async () => {
          const fd = new FormData();
          fd.append("file", input.files[0]);
          try {
            const r = await YS.api(`/api/datasets/${d.id}/import/zip`, { method: "POST", body: fd });
            this.showToast(`ZIP 导入 ${r.imported} 张，含标注 ${r.labeled}`);
            await this.loadDatasets();
          } catch (e) {
            this.showToast(e.message);
          }
        };
        input.click();
      },
      async uploadVideo(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*";
        input.onchange = async () => {
          const fd = new FormData();
          fd.append("file", input.files[0]);
          try {
            this.taskMsg[d.id] = "上传中…";
            const t = await YS.api(`/api/datasets/${d.id}/import/video`, { method: "POST", body: fd });
            // 接口返回后台任务，轮询进度
            if (t && t.id) this.pollTask(t.id, d.id);
            else {
              this.taskMsg[d.id] = `抽帧完成 ${t.frames || 0} 帧`;
              await this.loadDatasets();
            }
          } catch (e) {
            this.taskMsg[d.id] = e.message;
          }
        };
        input.click();
      },
      async pollTask(taskId, dsId) {
        for (let i = 0; i < 3600; i++) {
          const t = await YS.api(`/api/tasks/${taskId}`);
          this.taskMsg[dsId] = `${t.message || t.status} (${Math.round(t.progress || 0)}%)`;
          if (["success", "failed", "canceled"].includes(t.status)) {
            if (t.error) this.taskMsg[dsId] += " · " + t.error;
            await this.loadDatasets();
            return t;
          }
          await new Promise((r) => setTimeout(r, 1500));
        }
      },
      async openAutolabel(d) {
        this.alPanel.show = true;
        this.alPanel.ds = d;
        this.alPanel.model = "default";
        this.alPanel.conf = 0.15;
        this.alPanel.overwrite = false;
        this.alPanel.options = [];
        try {
          const r = await YS.api(`/api/datasets/${d.id}/autolabel-options`);
          this.alPanel.options = r.options || [];
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async submitAutolabel() {
        const d = this.alPanel.ds;
        if (!d) return;
        try {
          const t = await YS.api(`/api/datasets/${d.id}/autolabel`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: this.alPanel.model || "default",
              scope: "unlabeled",
              conf: this.alPanel.conf,
              overwrite: !!this.alPanel.overwrite,
            }),
          });
          this.alPanel.show = false;
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async clearAutoAnnotations(d) {
        if (
          !confirm(
            `将删除数据集「${d.name}」中所有模型自动标注的框（source=auto），你手工标的不受影响。确定？`
          )
        )
          return;
        try {
          const r = await YS.api(`/api/datasets/${d.id}/annotations/clear-auto`, {
            method: "POST",
          });
          this.showToast(
            `已删除 ${r.deleted_auto_boxes || 0} 个自动框，涉及 ${r.affected_images || 0} 张图`
          );
          await this.loadDatasets();
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async runOpenvocab(d) {
        const prompts = prompt(
          "开放词表提示词（逗号分隔）\n提示：CPU 上约 0.6~1.2 秒/张，大图集请先试 20 张",
          "motorcycle with front wheel lifted, three people on one motorcycle"
        );
        if (!prompts) return;
        const full = confirm(
          "点「确定」跑全量（CPU 约 0.6~1.2 秒/张，可能较久）；\n点「取消」只试 20 张（推荐先预览）"
        );
        try {
          const body = {
            prompts: prompts.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
            conf: 0.1,
          };
          if (!full) body.preview_limit = 20;
          const t = await YS.api(`/api/datasets/${d.id}/autolabel/openvocab`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          this.showToast(full ? "开放词表全量任务已提交（CPU 较慢）" : "开放词表试跑 20 张…");
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async editClasses(d) {
        const cur = (d.classes || []).join(", ");
        const text = prompt("编辑类别（逗号分隔，顺序很重要，改顺序会影响已有标注）", cur);
        if (text == null) return;
        const classes = text.split(/[,，;；]/).map((s) => s.trim()).filter(Boolean);
        if (!classes.length) {
          this.showToast("至少保留一个类别");
          return;
        }
        try {
          await YS.api(`/api/datasets/${d.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ classes }),
          });
          this.showToast("类别已更新");
          await this.loadDatasets();
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async runTrack(d) {
        // 需要起始图：取数据集第一张（或用户在标注页当前图）
        try {
          const r = await YS.api(`/api/datasets/${d.id}/images?page_size=1&sort=filename`);
          const start = (r.items || [])[0];
          if (!start) {
            this.showToast("数据集还没有图片，请先导入视频或图片");
            return;
          }
          const maxStr = prompt("从第一张起最多跟踪多少帧？（默认 300，上限 300）", "300");
          if (maxStr == null) return;
          const max_frames = Math.min(300, Math.max(1, parseInt(maxStr, 10) || 300));
          const t = await YS.api(`/api/datasets/${d.id}/track`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              start_image_id: start.id,
              max_frames,
              model: "default",
              conf: 0.25,
            }),
          });
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async runDedup(d) {
        try {
          const t = await YS.api(`/api/datasets/${d.id}/dedup`, { method: "POST" });
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      openCropImport(d) {
        this.cropPanel.show = true;
        this.cropPanel.dest = d;
        this.cropPanel.source = "zip";
        this.cropPanel.srcDatasetId = "";
        this.cropPanel.zipFile = null;
        this.cropPanel.busy = false;
        this.cropPanel.msg = "";
        this.cropPanel.taskId = null;
        this.cropPanel.preview = null;
      },
      closeCropImport() {
        this.cropPanel.show = false;
        this.cropPanel.busy = false;
      },
      onCropZipPick(e) {
        const f = e.target.files && e.target.files[0];
        this.cropPanel.zipFile = f || null;
      },
      cropParamsJson() {
        return JSON.stringify(this.cropPanel.params || {});
      },
      async runCropPreview() {
        const d = this.cropPanel.dest;
        if (!d) return;
        this.cropPanel.busy = true;
        this.cropPanel.msg = "预览中（CPU 检测大图较慢）…";
        this.cropPanel.preview = null;
        try {
          const fd = new FormData();
          fd.append("params", this.cropParamsJson());
          if (this.cropPanel.source === "zip") {
            if (!this.cropPanel.zipFile) throw new Error("请先选择大图 ZIP");
            fd.append("file", this.cropPanel.zipFile);
          } else {
            if (!this.cropPanel.srcDatasetId) throw new Error("请选择源数据集");
            fd.append("src_dataset_id", this.cropPanel.srcDatasetId);
          }
          const r = await YS.api(`/api/datasets/${d.id}/crop-import/preview`, {
            method: "POST",
            body: fd,
          });
          this.cropPanel.preview = r;
          this.cropPanel.msg = `预览完成：${r.crop_count || 0} 张切图`;
          if (!(r.items || []).length) {
            this.showToast("没有切出任何图，可调低 conf 或检查类别名是否为 person");
          }
        } catch (e) {
          this.cropPanel.msg = e.message;
          this.showToast(e.message);
        } finally {
          this.cropPanel.busy = false;
        }
      },
      async runCropFull() {
        const d = this.cropPanel.dest;
        if (!d) return;
        this.cropPanel.busy = true;
        this.cropPanel.msg = "提交全量切图任务…";
        try {
          let t;
          if (this.cropPanel.source === "zip") {
            if (!this.cropPanel.zipFile) throw new Error("请先选择大图 ZIP");
            const fd = new FormData();
            fd.append("file", this.cropPanel.zipFile);
            fd.append("params", this.cropParamsJson());
            t = await YS.api(`/api/datasets/${d.id}/crop-import/zip`, {
              method: "POST",
              body: fd,
            });
          } else {
            if (!this.cropPanel.srcDatasetId) throw new Error("请选择源数据集");
            t = await YS.api(
              `/api/datasets/${d.id}/crop-import/from/${this.cropPanel.srcDatasetId}`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this.cropPanel.params),
              }
            );
          }
          this.cropPanel.taskId = t.id;
          this.taskMsg[d.id] = "智能切图排队中…";
          this.pollTask(t.id, d.id).then((done) => {
            if (done && done.status === "success") {
              this.showToast(done.message || "切图完成，可以开始标注了");
              this.cropPanel.msg = done.message || "完成";
            }
          });
          this.cropPanel.msg = `任务已提交 ${t.id}，可关闭面板，进度在数据集卡片上查看`;
        } catch (e) {
          this.cropPanel.msg = e.message;
          this.showToast(e.message);
        } finally {
          this.cropPanel.busy = false;
        }
      },
      async cancelCropTask() {
        if (!this.cropPanel.taskId) return;
        try {
          await YS.api(`/api/tasks/${this.cropPanel.taskId}/cancel`, { method: "POST" });
          this.cropPanel.msg = "已请求取消";
          this.showToast("已取消切图任务（已入库的切图会保留）");
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async loadBaseModels() {
        this.baseModels = await YS.api("/api/models/base-options");
        this.onGroupChange();
      },
      onGroupChange() {
        const opts = this.baseModelOptions.filter((m) => m.available);
        if (opts.length) {
          if (!opts.find((m) => m.id === this.trainForm.base_model)) {
            this.trainForm.base_model = opts[0].id;
          }
        }
        if (this.trainForm.group === "finetune") this.trainForm.epochs = 20;
        else this.trainForm.epochs = 40;
        this.refreshEstimate();
      },
      async refreshEstimate() {
        if (!this.trainForm.dataset_id) return;
        try {
          this.estimate = await YS.api("/api/train/estimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              dataset_id: this.trainForm.dataset_id,
              base_model: this.trainForm.base_model,
              epochs: this.trainForm.epochs,
              imgsz: this.trainForm.imgsz,
              batch: this.trainForm.batch,
              freeze: this.trainForm.freeze,
              only_confirmed: this.trainForm.only_confirmed,
            }),
          });
        } catch (e) {
          this.estimate = { human: "—", tips: [e.message], seconds: 0, cache: "-", seconds_per_epoch: 0 };
        }
      },
      async startTrain() {
        try {
          const body = {
            dataset_id: this.trainForm.dataset_id,
            base_model: this.trainForm.base_model,
            epochs: this.trainForm.epochs,
            imgsz: this.trainForm.imgsz,
            batch: this.trainForm.batch,
            freeze: this.trainForm.freeze,
            only_confirmed: this.trainForm.only_confirmed,
            force_long: this.trainForm.force_long,
          };
          const j = await YS.api("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          this.showToast("训练已提交 " + j.id);
          await this.loadJobs();
          this.watchJob(j);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async loadJobs() {
        this.trainJobs = await YS.api("/api/train");
      },
      async cancelTrain(j) {
        await YS.api(`/api/train/${j.id}/cancel`, { method: "POST" });
        await this.loadJobs();
      },
      async resumeTrain(j) {
        try {
          await YS.api(`/api/train/${j.id}/resume`, { method: "POST" });
          this.showToast("已继续训练");
          await this.loadJobs();
          this.watchJob(j);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async publishTrain(j) {
        try {
          const m = await YS.api(`/api/train/${j.id}/publish`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
          this.showToast("已发布 " + m.name + "；OpenVINO 导出在后台进行");
          this.loadModels();
        } catch (e) {
          this.showToast(e.message);
        }
      },
      watchJob(j) {
        this.activeJob = j;
        this.trainLog = "";
        if (this._logEs) this._logEs.close();
        this._logEs = new EventSource(`/api/train/${j.id}/logs`);
        this._logEs.onmessage = (ev) => {
          this.trainLog += ev.data + "\n";
          this.$nextTick(() => {
            const el = this.$refs.logbox;
            if (el) el.scrollTop = el.scrollHeight;
          });
        };
        this._pollMetrics(j.id);
      },
      async _pollMetrics(jobId) {
        if (this._poll) clearInterval(this._poll);
        const tick = async () => {
          try {
            const r = await YS.api(`/api/train/${jobId}/metrics`);
            this._drawChart(r.series || []);
            await this.loadJobs();
            const j = this.trainJobs.find((x) => x.id === jobId);
            if (j) this.activeJob = j;
            if (j && !["running", "pending"].includes(j.status)) {
              clearInterval(this._poll);
              this._poll = null;
            }
          } catch (_) {}
        };
        tick();
        this._poll = setInterval(tick, 5000);
      },
      _drawChart(series) {
        const el = document.getElementById("train-chart");
        if (!el || typeof echarts === "undefined") return;
        if (!this._chart) this._chart = echarts.init(el);
        const epochs = series.map((s) => s.epoch);
        this._chart.setOption({
          backgroundColor: "transparent",
          tooltip: { trigger: "axis" },
          legend: {
            data: ["box_loss", "cls_loss", "mAP50", "mAP50-95"],
            textStyle: { color: "#94a3b8" },
          },
          xAxis: { type: "category", data: epochs, axisLabel: { color: "#94a3b8" } },
          yAxis: [
            { type: "value", name: "loss", axisLabel: { color: "#94a3b8" }, splitLine: { lineStyle: { color: "#334155" } } },
            { type: "value", name: "mAP", max: 1, axisLabel: { color: "#94a3b8" }, splitLine: { show: false } },
          ],
          series: [
            { name: "box_loss", type: "line", data: series.map((s) => s.box_loss), smooth: true },
            { name: "cls_loss", type: "line", data: series.map((s) => s.cls_loss), smooth: true },
            { name: "mAP50", type: "line", yAxisIndex: 1, data: series.map((s) => s.mAP50), smooth: true },
            {
              name: "mAP50-95",
              type: "line",
              yAxisIndex: 1,
              data: series.map((s) => s["mAP50-95"] != null ? s["mAP50-95"] : s.mAP50_95),
              smooth: true,
            },
          ],
        });
      },
      async loadModels() {
        this.modelList = await YS.api("/api/models");
      },
      async setDefaultModel(m) {
        await YS.api(`/api/models/${m.id}/set-default-autolabel`, { method: "POST" });
        this.showToast("已设为默认预标注模型");
        this.loadModels();
      },
      async evalModel(m) {
        const dsId = this.evalDatasetId || (this.currentDs && this.currentDs.id) || (this.datasets[0] && this.datasets[0].id);
        if (!dsId) {
          this.showToast("请先选择或创建一个数据集再评估");
          return;
        }
        try {
          this.showToast("评估中（CPU 可能要一会儿）…");
          const r = await YS.api(`/api/models/${m.id}/evaluate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: dsId, conf: 0.25, imgsz: 416 }),
          });
          this.showToast(
            `评估完成：P≈${r.precision_approx} R≈${r.recall_approx}（${r.note || "粗评"}）`
          );
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async delModel(m) {
        if (!confirm("确定删除模型 " + m.name + "？")) return;
        await YS.api(`/api/models/${m.id}`, { method: "DELETE" });
        this.loadModels();
      },
    },
    async mounted() {
      try {
        this.sysInfo = await YS.api("/api/system/info");
      } catch (_) {}
      await this.loadDatasets();
    },
  }).mount("#app");
});
