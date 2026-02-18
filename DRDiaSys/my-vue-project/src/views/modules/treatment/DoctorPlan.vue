<template>
  <div class="doctor-plan-page" v-if="isDoctor">
    <el-row :gutter="20">
      <!-- 左侧：病例列表 -->
      <el-col :span="6">
        <el-card shadow="never" class="sidebar-card">
          <div class="sidebar-header">
            <h3>病例列表</h3>
            <el-button type="primary" size="small" @click="fetchCases">
              <el-icon><RefreshRight /></el-icon>
              刷新
            </el-button>
          </div>
          <el-input
            v-model="caseSearchQuery"
            placeholder="搜索患者"
            clearable
            class="search-input"
            @input="filterCases"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <el-table
            :data="filteredCases"
            height="calc(100vh - 200px)"
            v-loading="caseLoading"
            @row-click="handleCaseSelect"
            :row-class-name="rowClassName"
          >
            <el-table-column prop="title" label="病例" min-width="120" />
            <el-table-column prop="patient_name" label="患者" width="100" />
          </el-table>
        </el-card>
      </el-col>

      <!-- 右侧：治疗方案管理 -->
      <el-col :span="18">
        <el-card shadow="never" v-if="selectedCase">
          <template #header>
            <div class="card-header">
              <div>
                <h3>{{ selectedCase.title }}</h3>
                <p class="patient-info">患者：{{ selectedCase.patient_name }}</p>
              </div>
              <div class="header-actions">
                <el-button type="success" @click="openCreateDialog">
                  <el-icon><Plus /></el-icon>
                  新建方案
                </el-button>
              </div>
            </div>
          </template>

          <!-- 方案列表 -->
          <el-tabs v-model="activeTab">
            <el-tab-pane label="全部" name="all">
              <PlanList
                :key="planListKey"
                :case-id="selectedCase.id"
                :is-doctor="true"
                @edit="handleEdit"
                @delete="handleDelete"
                @confirm="handleConfirm"
                @complete="handleComplete"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="草稿" name="draft">
              <PlanList
                :key="planListKey"
                :case-id="selectedCase.id"
                :is-doctor="true"
                status="draft"
                @edit="handleEdit"
                @confirm="handleConfirm"
                @delete="handleDelete"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="已确认" name="confirmed">
              <PlanList
                :key="planListKey"
                :case-id="selectedCase.id"
                :is-doctor="true"
                status="confirmed"
                @edit="handleEdit"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="进行中" name="active">
              <PlanList
                :key="planListKey"
                :case-id="selectedCase.id"
                :is-doctor="true"
                status="active"
                @edit="handleEdit"
                @confirm="handleConfirm"
                @complete="handleComplete"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="已完成" name="completed">
              <PlanList
                :key="planListKey"
                :case-id="selectedCase.id"
                :is-doctor="true"
                status="completed"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
          </el-tabs>
        </el-card>
        <el-card v-else shadow="never">
          <el-empty description="请选择一个病例" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 创建/编辑方案对话框 -->
    <el-dialog
      v-model="planDialogVisible"
      :title="editingPlan ? '编辑治疗方案' : '新建治疗方案'"
      width="900px"
      top="5vh"
      :close-on-click-modal="false"
    >
      <el-form :model="planForm" label-width="120px" ref="planFormRef">
        <!-- 基本信息 -->
        <el-divider content-position="left">基本信息</el-divider>
        <el-form-item label="方案标题" required>
          <el-input v-model="planForm.title" placeholder="例如：DR2级中度NPDR治疗方案" />
        </el-form-item>

        <!-- 一、基础管理目标 -->
        <el-divider content-position="left">一、基础管理目标</el-divider>
        <div class="form-section">
          <h4>血糖控制目标</h4>
          <div class="target-group">
            <label class="target-label">空腹血糖</label>
            <div class="range-input">
              <el-input v-model="planForm.blood_sugar_target.fasting_min" placeholder="最小值" class="range-input-min" />
              <span class="range-separator">—</span>
              <el-input v-model="planForm.blood_sugar_target.fasting_max" placeholder="最大值" class="range-input-max" />
              <el-select v-model="planForm.blood_sugar_target.fasting_unit" class="unit-select">
                <el-option label="mmol/L" value="mmol/L" />
                <el-option label="mg/dL" value="mg/dL" />
              </el-select>
            </div>
          </div>
          <div class="target-group">
            <label class="target-label">餐后血糖</label>
            <div class="range-input">
              <el-input v-model="planForm.blood_sugar_target.postprandial_min" placeholder="最小值" class="range-input-min" />
              <span class="range-separator">—</span>
              <el-input v-model="planForm.blood_sugar_target.postprandial_max" placeholder="最大值" class="range-input-max" />
              <el-select v-model="planForm.blood_sugar_target.postprandial_unit" class="unit-select">
                <el-option label="mmol/L" value="mmol/L" />
                <el-option label="mg/dL" value="mg/dL" />
              </el-select>
            </div>
          </div>
          <div class="target-group">
            <label class="target-label">HbA1c</label>
            <div class="range-input">
              <el-input v-model="planForm.blood_sugar_target.hba1c_min" placeholder="最小值" class="range-input-min" />
              <span class="range-separator">—</span>
              <el-input v-model="planForm.blood_sugar_target.hba1c_max" placeholder="最大值" class="range-input-max" />
              <el-select v-model="planForm.blood_sugar_target.hba1c_unit" class="unit-select">
                <el-option label="%" value="%" />
                <el-option label="mmol/mol" value="mmol/mol" />
              </el-select>
            </div>
          </div>

          <h4>血压控制目标</h4>
          <div class="target-group">
            <label class="target-label">收缩压</label>
            <div class="range-input">
              <el-input v-model="planForm.blood_pressure_target.systolic_min" placeholder="最小值" class="range-input-min" />
              <span class="range-separator">—</span>
              <el-input v-model="planForm.blood_pressure_target.systolic_max" placeholder="最大值" class="range-input-max" />
              <el-select v-model="planForm.blood_pressure_target.systolic_unit" class="unit-select">
                <el-option label="mmHg" value="mmHg" />
                <el-option label="kPa" value="kPa" />
              </el-select>
            </div>
          </div>
          <div class="target-group">
            <label class="target-label">舒张压</label>
            <div class="range-input">
              <el-input v-model="planForm.blood_pressure_target.diastolic_min" placeholder="最小值" class="range-input-min" />
              <span class="range-separator">—</span>
              <el-input v-model="planForm.blood_pressure_target.diastolic_max" placeholder="最大值" class="range-input-max" />
              <el-select v-model="planForm.blood_pressure_target.diastolic_unit" class="unit-select">
                <el-option label="mmHg" value="mmHg" />
                <el-option label="kPa" value="kPa" />
              </el-select>
            </div>
          </div>

          <div class="target-group" style="margin-top: 15px;">
            <label class="target-label">血脂管理</label>
            <el-input
              v-model="planForm.lipid_management"
              type="textarea"
              :rows="2"
              placeholder="如：LDL-C < 2.6 mmol/L，推荐使用他汀类药物"
              class="text-input"
            />
          </div>
        </div>

        <!-- 二、治疗方案 -->
        <el-divider content-position="left">二、治疗方案</el-divider>
        <el-form-item label="治疗项目">
          <el-table :data="planForm.treatments" border size="small">
            <el-table-column label="治疗类别" width="110">
              <template #default="scope">
                <el-select
                  v-model="scope.row.category"
                  placeholder="选择"
                  filterable
                  allow-create
                  default-first-option
                  size="small"
                  @change="handleTreatmentCategoryChange(scope.row)"
                  style="width: 100%"
                >
                  <el-option label="抗VEGF" value="anti_vegf" />
                  <el-option label="激光治疗" value="laser" />
                  <el-option label="手术治疗" value="surgical" />
                  <el-option label="其他治疗" value="other" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="具体项目" width="200">
              <template #default="scope">
                <el-select
                  v-model="scope.row.item"
                  :placeholder="getTreatmentPlaceholder(scope.row.category, 'item')"
                  filterable
                  allow-create
                  default-first-option
                  size="small"
                  style="width: 100%"
                >
                  <el-option
                    v-for="opt in getTreatmentOptions(scope.row.category)"
                    :key="opt.value"
                    :label="opt.label"
                    :value="opt.value"
                  />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="频率" width="140">
              <template #default="scope">
                <el-input v-model="scope.row.frequency" placeholder="如：每月" size="small" style="width: 70px" />
                <el-select v-model="scope.row.frequency_unit" size="small" style="width: 60px; margin-left: 5px">
                  <el-option label="次/天" value="次/天" />
                  <el-option label="次/周" value="次/周" />
                  <el-option label="次/月" value="次/月" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="疗程" width="140">
              <template #default="scope">
                <el-input v-model="scope.row.course" placeholder="如：3-5" size="small" style="width: 60px" />
                <el-select v-model="scope.row.course_unit" size="small" style="width: 70px; margin-left: 5px">
                  <el-option label="天" value="天" />
                  <el-option label="周" value="周" />
                  <el-option label="月" value="月" />
                  <el-option label="次" value="次" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="备注/适应症">
              <template #default="scope">
                <el-input
                  v-model="scope.row.notes"
                  :placeholder="getTreatmentPlaceholder(scope.row.category, 'notes')"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column label="操作" width="60">
              <template #default="scope">
                <el-button
                  type="danger"
                  link
                  size="small"
                  @click="removeTreatment(scope.$index)"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-button type="primary" link @click="addTreatment" style="margin-top: 10px">
            <el-icon><Plus /></el-icon>
            添加治疗项目
          </el-button>
        </el-form-item>

        <!-- 三、药物治疗 -->
        <el-divider content-position="left">三、药物治疗</el-divider>
        <el-form-item label="药物列表">
          <el-table :data="planForm.medications" border size="small">
            <el-table-column label="类别" width="100">
              <template #default="scope">
                <el-select
                  v-model="scope.row.category"
                  placeholder="选择"
                  filterable
                  allow-create
                  default-first-option
                  size="small"
                  style="width: 100%"
                >
                  <el-option label="眼科用药" value="ophthalmic" />
                  <el-option label="全身用药" value="systemic" />
                  <el-option label="注射用药" value="injection" />
                  <el-option label="营养补充" value="nutrient" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="药物名称" width="140">
              <template #default="scope">
                <el-select
                  v-model="scope.row.name"
                  placeholder="搜索或输入"
                  filterable
                  allow-create
                  default-first-option
                  size="small"
                  style="width: 100%"
                >
                  <el-option-group
                    v-for="group in getMedicationOptionsByCategory(scope.row.category)"
                    :key="group.label"
                    :label="group.label"
                  >
                    <el-option
                      v-for="opt in group.options"
                      :key="opt.value"
                      :label="opt.label"
                      :value="opt.value"
                    />
                  </el-option-group>
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="给药途径" width="100">
              <template #default="scope">
                <el-select
                  v-model="scope.row.route"
                  placeholder="途径"
                  size="small"
                  style="width: 100%"
                >
                  <el-option label="口服" value="口服" />
                  <el-option label="外用" value="外用" />
                  <el-option label="滴眼" value="滴眼" />
                  <el-option label="注射" value="注射" />
                  <el-option label="舌下含服" value="舌下含服" />
                  <el-option label="吸入" value="吸入" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="单次剂量" width="180">
              <template #default="scope">
                <div style="display: flex; align-items: center; gap: 4px;">
                  <el-input-number v-model="scope.row.dose_value" :min="0" :precision="2" size="small" style="width: 90px" />
                  <el-select v-model="scope.row.dose_unit" size="small" style="width: 70px">
                    <el-option label="mg" value="mg" />
                    <el-option label="g" value="g" />
                    <el-option label="ml" value="ml" />
                    <el-option label="μg" value="μg" />
                    <el-option label="片" value="片" />
                    <el-option label="粒" value="粒" />
                    <el-option label="支" value="支" />
                    <el-option label="U" value="U" />
                    <el-option label="滴" value="滴" />
                  </el-select>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="给药频次" width="100">
              <template #default="scope">
                <el-select
                  v-model="scope.row.frequency"
                  placeholder="频次"
                  size="small"
                  style="width: 100%"
                >
                  <el-option label="每日1次" value="每日1次" />
                  <el-option label="每日2次" value="每日2次" />
                  <el-option label="每日3次" value="每日3次" />
                  <el-option label="每日4次" value="每日4次" />
                  <el-option label="每晚1次" value="每晚1次" />
                  <el-option label="必要时" value="必要时" />
                  <el-option label="餐前" value="餐前" />
                  <el-option label="餐后" value="餐后" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="疗程" width="160">
              <template #default="scope">
                <div style="display: flex; align-items: center; gap: 4px;">
                  <el-input-number v-model="scope.row.duration_value" :min="0" size="small" style="width: 80px" />
                  <el-select v-model="scope.row.duration_unit" size="small" style="width: 65px">
                    <el-option label="天" value="天" />
                    <el-option label="周" value="周" />
                    <el-option label="月" value="月" />
                  </el-select>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="备注" width="100">
              <template #default="scope">
                <el-input v-model="scope.row.notes" placeholder="备注" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="操作" width="60">
              <template #default="scope">
                <el-button
                  type="danger"
                  link
                  size="small"
                  @click="removeMedication(scope.$index)"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-button type="primary" link @click="addMedication" style="margin-top: 10px">
            <el-icon><Plus /></el-icon>
            添加药物
          </el-button>
        </el-form-item>

        <!-- 四、生活方式干预 -->
        <el-divider content-position="left">四、生活方式干预</el-divider>
        <el-form-item label="饮食指导">
          <el-input
            v-model="planForm.diet_guidance"
            type="textarea"
            :rows="4"
            placeholder="• 控制总热量摄入&#10;• 减少精制碳水化合物&#10;• 增加深海鱼类摄入..."
          />
        </el-form-item>
        <el-form-item label="运动指导">
          <el-input
            v-model="planForm.exercise_guidance"
            type="textarea"
            :rows="4"
            placeholder="• 每周至少150分钟中等强度有氧运动&#10;• 餐后1小时进行..."
          />
        </el-form-item>
        <el-form-item label="综合建议">
          <el-input
            v-model="planForm.lifestyle_advice"
            type="textarea"
            :rows="3"
            placeholder="定期监测血糖、血压、血脂，每年进行一次全面眼科检查..."
          />
        </el-form-item>

        <!-- 五、随访监测 -->
        <el-divider content-position="left">五、随访监测</el-divider>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="复查间隔">
              <el-input-number
                v-model="planForm.follow_up_plan.interval_days"
                :min="30"
                :max="365"
                :step="30"
              />
              <span class="unit">天</span>
            </el-form-item>
          </el-col>
          <el-col :span="16">
            <el-form-item label="检查项目">
              <el-input
                v-model="planForm.follow_up_plan.check_items_text"
                placeholder="用逗号分隔，如：眼底检查、视力检查、OCT"
              />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="复查日期">
          <el-date-picker
            v-model="planForm.follow_up_plan.next_date"
            type="date"
            placeholder="选择下次复查日期"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="预警症状">
          <el-input
            v-model="planForm.warning_symptoms"
            type="textarea"
            :rows="3"
            placeholder="如：视力突然下降、眼前大量黑影漂浮、红色幕布样遮挡..."
          />
        </el-form-item>

        <!-- 六、注意事项 -->
        <el-divider content-position="left">六、注意事项</el-divider>
        <el-form-item label="注意事项">
          <el-input
            v-model="planForm.precautions"
            type="textarea"
            :rows="3"
            placeholder="用药注意事项、风险提示等"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="planDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="savePlan" :loading="saving">保存</el-button>
      </template>
    </el-dialog>

    <!-- 执行记录对话框 -->
    <el-dialog
      v-model="executionDialogVisible"
      title="方案执行记录"
      width="700px"
    >
      <ExecutionList 
        :plan-id="selectedPlanId" 
        :medications="selectedPlan?.medications || []"
        :plan-start-date="selectedPlan?.created_at ? new Date(selectedPlan.created_at).toISOString().split('T')[0] : null"
      />
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage, ElMessageBox } from 'element-plus'
import { RefreshRight, Search, Plus } from '@element-plus/icons-vue'
import PlanList from './components/PlanList.vue'
import ExecutionList from './components/ExecutionList.vue'

export default {
  name: 'DoctorPlan',
  components: {
    RefreshRight,
    Search,
    Plus,
    PlanList,
    ExecutionList
  },
  data() {
    return {
      isDoctor: localStorage.getItem('userRole') === 'doctor',
      caseLoading: false,
      cases: [],
      filteredCases: [],
      caseSearchQuery: '',
      selectedCase: null,
      activeTab: 'draft',
      planListKey: 0,
      planDialogVisible: false,
      editingPlan: null,
      planForm: this.getEmptyForm(),
      planFormRef: null,
      saving: false,
      executionDialogVisible: false,
      selectedPlanId: null,
      selectedPlan: null,
      // 常见药物选项（不包含治疗项目中的注射药物，避免重复）
      medicationOptions: [
        {
          label: '糖尿病用药',
          options: [
            { value: '二甲双胍', label: '二甲双胍' },
            { value: '格列美脲', label: '格列美脲' },
            { value: '格列齐特', label: '格列齐特' },
            { value: '格列吡嗪', label: '格列吡嗪' },
            { value: '阿卡波糖', label: '阿卡波糖' },
            { value: '吡格列酮', label: '吡格列酮' },
            { value: '西格列汀', label: '西格列汀' },
            { value: '达格列净', label: '达格列净' },
            { value: '恩格列净', label: '恩格列净' },
            { value: '利拉鲁肽', label: '利拉鲁肽' },
            { value: '司美格鲁肽', label: '司美格鲁肽' },
            { value: '甘精胰岛素', label: '甘精胰岛素' },
            { value: '门冬胰岛素', label: '门冬胰岛素' },
            { value: '地特胰岛素', label: '地特胰岛素' }
          ]
        },
        {
          label: '降压药',
          options: [
            { value: '氨氯地平', label: '氨氯地平' },
            { value: '硝苯地平', label: '硝苯地平' },
            { value: '非洛地平', label: '非洛地平' },
            { value: '贝那普利', label: '贝那普利' },
            { value: '培哚普利', label: '培哚普利' },
            { value: '缬沙坦', label: '缬沙坦' },
            { value: '厄贝沙坦', label: '厄贝沙坦' },
            { value: '美托洛尔', label: '美托洛尔' },
            { value: '比索洛尔', label: '比索洛尔' }
          ]
        },
        {
          label: '降脂药',
          options: [
            { value: '阿托伐他汀', label: '阿托伐他汀' },
            { value: '瑞舒伐他汀', label: '瑞舒伐他汀' },
            { value: '辛伐他汀', label: '辛伐他汀' },
            { value: '非诺贝特', label: '非诺贝特' }
          ]
        },
        {
          label: '营养补充',
          options: [
            { value: '叶酸', label: '叶酸' },
            { value: '维生素B族', label: '维生素B族' },
            { value: '维生素C', label: '维生素C' },
            { value: '维生素E', label: '维生素E' },
            { value: '鱼油', label: '鱼油' },
            { value: '亚麻酸', label: '亚麻酸' },
            { value: '硫辛酸', label: '硫辛酸' }
          ]
        },
        {
          label: '辅助用药',
          options: [
            { value: '甲钴胺片', label: '甲钴胺片' },
            { value: '羟苯磺酸钙', label: '羟苯磺酸钙' },
            { value: '递法明片', label: '递法明片' },
            { value: '复方樟柳碱注射液', label: '复方樟柳碱注射液' }
          ]
        }
      ]
    }
  },
  created() {
    this.fetchCases()
  },
  methods: {
    getEmptyForm() {
      return {
        title: '',
        // 基础管理目标
        blood_sugar_target: {
          fasting_min: '',
          fasting_max: '',
          fasting_unit: 'mmol/L',
          postprandial_min: '',
          postprandial_max: '',
          postprandial_unit: 'mmol/L',
          hba1c_min: '',
          hba1c_max: '',
          hba1c_unit: '%'
        },
        blood_pressure_target: {
          systolic_min: '',
          systolic_max: '',
          systolic_unit: 'mmHg',
          diastolic_min: '',
          diastolic_max: '',
          diastolic_unit: 'mmHg'
        },
        lipid_management: '',
        // 治疗方案（动态列表）
        treatments: [],
        // 药物治疗
        medications: [],
        // 生活方式
        diet_guidance: '',
        exercise_guidance: '',
        lifestyle_advice: '',
        // 随访监测
        follow_up_plan: {
          next_date: null,
          check_items_text: '',
          interval_days: 90
        },
        warning_symptoms: '',
        // 注意事项
        precautions: ''
      }
    },
    addTreatment() {
      this.planForm.treatments.push({
        category: '',
        item: '',
        frequency: '',
        frequency_unit: '次',
        course: '',
        course_unit: '月',
        notes: ''
      })
    },
    removeTreatment(index) {
      this.planForm.treatments.splice(index, 1)
    },
    handleTreatmentCategoryChange(row) {
      // 清空当前行的值，便于用户重新输入
      row.item = ''
      row.frequency = ''
      row.course = ''
      row.notes = ''
    },
    getTreatmentPlaceholder(category, field) {
      const placeholders = {
        anti_vegf: {
          item: '如：雷珠单抗、康柏西普、阿柏西普',
          frequency: '如：每月1次',
          course: '如：3-5次为一个疗程',
          notes: '如：合并DME或高危特征'
        },
        laser: {
          item: '如：全视网膜光凝(PRP)',
          frequency: '如：分3-4次完成',
          course: '如：间隔1-2周',
          notes: '如：避免一次性激光量过大'
        },
        surgical: {
          item: '如：玻璃体切割术',
          frequency: '如：一次性',
          course: '如：根据恢复情况',
          notes: '如：指征：玻璃体出血不吸收'
        },
        other: {
          item: '请输入具体治疗项目',
          frequency: '请输入频率',
          course: '请输入疗程',
          notes: '请输入备注或适应症'
        }
      }
      return placeholders[category]?.[field] || ''
    },
    getTreatmentOptions(category) {
      const options = {
        anti_vegf: [
          { value: '雷珠单抗', label: '雷珠单抗' },
          { value: '康柏西普', label: '康柏西普' },
          { value: '阿柏西普', label: '阿柏西普' },
          { value: '贝伐单抗', label: '贝伐单抗' },
          { value: '布西珠单抗', label: '布西珠单抗' }
        ],
        laser: [
          { value: '全视网膜光凝(PRP)', label: '全视网膜光凝(PRP)' },
          { value: '局灶光凝', label: '局灶光凝' },
          { value: '格栅样光凝', label: '格栅样光凝' },
          { value: '黄斑光凝', label: '黄斑光凝' },
          { value: 'TTT(经瞳孔温热疗法)', label: 'TTT(经瞳孔温热疗法)' },
          { value: 'PDT(光动力疗法)', label: 'PDT(光动力疗法)' }
        ],
        surgical: [
          { value: '玻璃体切割术', label: '玻璃体切割术' },
          { value: '白内障超声乳化手术', label: '白内障超声乳化手术' },
          { value: '人工晶状体植入术', label: '人工晶状体植入术' },
          { value: '视网膜复位术', label: '视网膜复位术' },
          { value: '巩膜扣带术', label: '巩膜扣带术' },
          { value: '睫状体冷冻术', label: '睫状体冷冻术' }
        ],
        other: []
      }
      return options[category] || []
    },
    getMedicationOptionsByCategory(category) {
      // 根据类别返回对应的药物选项，如果没有选择类别则显示所有
      if (!category) {
        return this.medicationOptions
      }
      
      // 映射类别到 medicationOptions 中的索引
      const categoryMap = {
        'ophthalmic': 4,      // 眼科用药 -> 辅助用药（甲钴胺、羟苯磺酸钙等）
        'systemic': [0, 1, 2], // 全身用药 -> 糖尿病用药 + 降压药 + 降脂药
        'injection': [],       // 注射用药 -> 无（都在治疗项目中）
        'nutrient': 3         // 营养补充 -> 营养补充
      }
      
      const mapping = categoryMap[category]
      if (Array.isArray(mapping)) {
        return mapping.map(index => this.medicationOptions[index])
      } else if (mapping !== undefined) {
        return [this.medicationOptions[mapping]]
      }
      return this.medicationOptions
    },
    async fetchCases() {
      this.caseLoading = true
      try {
        const res = await api.get('/api/diagnosis/cases/')
        this.cases = res.data
        this.filteredCases = res.data
      } catch (error) {
        ElMessage.error('获取病例列表失败')
      } finally {
        this.caseLoading = false
      }
    },
    filterCases() {
      if (!this.caseSearchQuery) {
        this.filteredCases = this.cases
        return
      }
      this.filteredCases = this.cases.filter(caseItem =>
        caseItem.title.includes(this.caseSearchQuery) ||
        caseItem.patient_name.includes(this.caseSearchQuery)
      )
    },
    handleCaseSelect(row) {
      this.selectedCase = row
    },
    rowClassName({ row }) {
      return row.id === this.selectedCase?.id ? 'selected-row' : ''
    },
    openCreateDialog() {
      if (!this.selectedCase) {
        ElMessage.warning('请先选择病例')
        return
      }
      this.editingPlan = null
      this.planForm = this.getEmptyForm()
      this.planDialogVisible = true
    },
    addMedication() {
      this.planForm.medications.push({
        name: '',
        route: '口服',
        dose_value: null,
        dose_unit: 'mg',
        frequency: '每日3次',
        duration_value: null,
        duration_unit: '月',
        category: 'ophthalmic',
        notes: ''
      })
    },
    removeMedication(index) {
      this.planForm.medications.splice(index, 1)
    },
    async savePlan() {
      if (!this.planForm.title) {
        ElMessage.warning('请输入方案标题')
        return
      }
      this.saving = true
      try {
        // 构建完整的数据对象
        const data = {
          case: this.selectedCase.id,
          title: this.planForm.title,
          // 基础管理目标（范围格式）
          blood_sugar_target: {
            fasting_min: this.planForm.blood_sugar_target.fasting_min,
            fasting_max: this.planForm.blood_sugar_target.fasting_max,
            fasting_unit: this.planForm.blood_sugar_target.fasting_unit,
            postprandial_min: this.planForm.blood_sugar_target.postprandial_min,
            postprandial_max: this.planForm.blood_sugar_target.postprandial_max,
            postprandial_unit: this.planForm.blood_sugar_target.postprandial_unit,
            hba1c_min: this.planForm.blood_sugar_target.hba1c_min,
            hba1c_max: this.planForm.blood_sugar_target.hba1c_max,
            hba1c_unit: this.planForm.blood_sugar_target.hba1c_unit
          },
          blood_pressure_target: {
            systolic_min: this.planForm.blood_pressure_target.systolic_min,
            systolic_max: this.planForm.blood_pressure_target.systolic_max,
            systolic_unit: this.planForm.blood_pressure_target.systolic_unit,
            diastolic_min: this.planForm.blood_pressure_target.diastolic_min,
            diastolic_max: this.planForm.blood_pressure_target.diastolic_max,
            diastolic_unit: this.planForm.blood_pressure_target.diastolic_unit
          },
          lipid_management: this.planForm.lipid_management,
          // 治疗方案（动态列表）
          treatments: this.planForm.treatments.map(t => ({
            category: t.category,
            item: t.item,
            frequency: t.frequency,
            frequency_unit: t.frequency_unit,
            course: t.course,
            course_unit: t.course_unit,
            notes: t.notes
          })),
          // 药物治疗
          medications: this.planForm.medications.map(m => ({
            name: m.name,
            route: m.route || '口服',
            // 合并给药途径和频次到 dosage 字段
            dosage: m.route && m.frequency ? `${m.route} ${m.frequency}` : (m.route || ''),
            dose: m.dose_value && m.dose_unit ? `${m.dose_value}${m.dose_unit}` : '',
            dose_value: m.dose_value,
            dose_unit: m.dose_unit,
            frequency: m.frequency || '',
            duration: m.duration_value && m.duration_unit ? `${m.duration_value}${m.duration_unit}` : '',
            duration_value: m.duration_value,
            duration_unit: m.duration_unit,
            category: m.category,
            notes: m.notes
          })),
          // 生活方式
          diet_guidance: this.planForm.diet_guidance,
          exercise_guidance: this.planForm.exercise_guidance,
          lifestyle_advice: this.planForm.lifestyle_advice,
          // 随访监测
          follow_up_plan: {
            next_date: this.planForm.follow_up_plan.next_date,
            check_items: this.planForm.follow_up_plan.check_items_text
              .split(',').map(s => s.trim()).filter(s => s),
            interval_days: this.planForm.follow_up_plan.interval_days
          },
          warning_symptoms: this.planForm.warning_symptoms,
          // 注意事项
          precautions: this.planForm.precautions
        }

        if (this.editingPlan) {
          await api.patch(`/api/treatment/plans/${this.editingPlan.id}/`, data)
          ElMessage.success('方案更新成功')
        } else {
          const res = await api.post('/api/treatment/plans/', data)
          ElMessage.success('方案创建成功')

          // 自动创建病历事件记录
          try {
            await api.post(`/api/diagnosis/cases/${this.selectedCase.id}/events/`, {
              event_type: 'treatment',
              description: `创建治疗方案：${res.data.title}（方案编号：${res.data.plan_number}）`,
              related_report_id: null,
              related_plan_id: res.data.id,
              next_followup_date: null
            })
          } catch (eventError) {
            console.error('创建病历事件记录失败:', eventError)
          }
        }
        this.planDialogVisible = false
        this.planListKey++
      } catch (error) {
        ElMessage.error('保存方案失败')
        console.error(error)
      } finally {
        this.saving = false
      }
    },
    handleEdit(plan) {
      this.editingPlan = plan
      this.planForm = {
        title: plan.title || '',
        // 基础管理目标（范围格式）
        blood_sugar_target: {
          fasting_min: plan.blood_sugar_target?.fasting_min || '',
          fasting_max: plan.blood_sugar_target?.fasting_max || '',
          fasting_unit: plan.blood_sugar_target?.fasting_unit || 'mmol/L',
          postprandial_min: plan.blood_sugar_target?.postprandial_min || '',
          postprandial_max: plan.blood_sugar_target?.postprandial_max || '',
          postprandial_unit: plan.blood_sugar_target?.postprandial_unit || 'mmol/L',
          hba1c_min: plan.blood_sugar_target?.hba1c_min || '',
          hba1c_max: plan.blood_sugar_target?.hba1c_max || '',
          hba1c_unit: plan.blood_sugar_target?.hba1c_unit || '%'
        },
        blood_pressure_target: {
          systolic_min: plan.blood_pressure_target?.systolic_min || '',
          systolic_max: plan.blood_pressure_target?.systolic_max || '',
          systolic_unit: plan.blood_pressure_target?.systolic_unit || 'mmHg',
          diastolic_min: plan.blood_pressure_target?.diastolic_min || '',
          diastolic_max: plan.blood_pressure_target?.diastolic_max || '',
          diastolic_unit: plan.blood_pressure_target?.diastolic_unit || 'mmHg'
        },
        lipid_management: plan.lipid_management || '',
        // 治疗方案（动态列表）
        treatments: (plan.treatments || []).map(t => ({
          category: t.category || '',
          item: t.item || '',
          frequency: t.frequency || '',
          frequency_unit: t.frequency_unit || '次',
          course: t.course || '',
          course_unit: t.course_unit || '月',
          notes: t.notes || ''
        })),
        // 药物治疗
        medications: (plan.medications || []).map(m => {
          // 解析给药途径和频次（旧数据格式兼容）
          let route = '口服'
          let frequency = '每日3次'
          let dose_value = null
          let dose_unit = 'mg'
          
          // 频次代码映射表（兼容旧数据）
          const frequencyMap = {
            'qd': '每日1次',
            'bid': '每日2次',
            'tid': '每日3次',
            'qid': '每日4次',
            'qn': '每晚1次',
            'prn': '必要时',
            'ac': '餐前',
            'pc': '餐后'
          }
          
          // 如果有新字段，直接使用
          if (m.route) {
            route = m.route
          }
          if (m.frequency) {
            // 如果是旧代码，转换为中文
            frequency = frequencyMap[m.frequency] || m.frequency
          }
          
          // 解析剂量
          if (m.dose) {
            const match = m.dose.match(/^([\d.]+)?([^\d\s]*)?$/)
            if (match) {
              dose_value = match[1] ? parseFloat(match[1]) : null
              dose_unit = match[2] || 'mg'
            }
          }
          
          // 兼容旧数据：从 dosage 中解析
          if (!m.route && m.dosage) {
            const match = m.dosage.match(/^([^\d\s]+)?\s*([a-zA-Z]+)?$/)
            if (match) {
              if (match[1]) route = match[1].trim()
              if (match[2]) {
                const engFreq = match[2].trim()
                frequency = frequencyMap[engFreq] || engFreq
              }
            }
          }
          
          // 解析剂量（旧数据格式）
          if (!m.dose && m.dosage_value) {
            dose_value = m.dosage_value
            dose_unit = m.dosage_unit || 'mg'
          }
          
          // 兼容旧数据：从 dosage 中解析剂量数字
          if (!m.dose && !m.dosage_value && m.dosage) {
            const match = m.dosage.match(/^([\d.]+)?([^\d\s]+)?\s*(.*)$/)
            if (match) {
              dose_value = match[1] ? parseFloat(match[1]) : null
              dose_unit = match[2] || 'mg'
              // 如果第三个捕获组是频次代码
              if (match[3] && !m.frequency && !m.dosage_frequency) {
                const engFreq = match[3].trim()
                frequency = frequencyMap[engFreq] || engFreq
              }
            }
          }
          
          // 兼容旧数据
          if (m.dosage_frequency) {
            frequency = frequencyMap[m.dosage_frequency] || m.dosage_frequency
          }
          
          // 解析疗程
          let duration_value = null
          let duration_unit = '月'
          if (m.duration) {
            const match = m.duration.match(/^([\d.]+)?([^\d\s]*)?$/)
            if (match) {
              duration_value = match[1] ? parseFloat(match[1]) : null
              duration_unit = match[2] || '月'
            }
          }
          if (m.duration_value) {
            duration_value = m.duration_value
            duration_unit = m.duration_unit || '月'
          }
          
          return {
            name: m.name || '',
            route: route,
            frequency: frequency,
            dose_value: dose_value,
            dose_unit: dose_unit,
            duration_value: duration_value,
            duration_unit: duration_unit,
            category: m.category || 'ophthalmic',
            notes: m.notes || ''
          }
        }),
        // 生活方式
        diet_guidance: plan.diet_guidance || '',
        exercise_guidance: plan.exercise_guidance || '',
        lifestyle_advice: plan.lifestyle_advice || '',
        // 随访监测
        follow_up_plan: {
          next_date: plan.follow_up_plan?.next_date || null,
          check_items_text: plan.follow_up_plan?.check_items?.join(',') || '',
          interval_days: plan.follow_up_plan?.interval_days || 90
        },
        warning_symptoms: plan.warning_symptoms || '',
        precautions: plan.precautions || ''
      }
      this.planDialogVisible = true
    },
    async handleConfirm(plan) {
      try {
        await ElMessageBox.confirm('确认后方案将通知患者，是否继续？', '确认方案', {
          type: 'warning'
        })
        await api.post(`/api/treatment/plans/${plan.id}/confirm/`)
        ElMessage.success('方案已确认')
        this.planListKey++
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('确认方案失败')
        }
      }
    },
    async handleComplete(plan) {
      try {
        await ElMessageBox.confirm('完成后方案将移至已完成列表，是否继续？', '完成方案', {
          type: 'warning'
        })
        await api.post(`/api/treatment/plans/${plan.id}/complete/`)
        ElMessage.success('方案已完成')
        this.planListKey++
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('完成方案失败')
        }
      }
    },
    async handleDelete(plan) {
      try {
        await ElMessageBox.confirm(
          `确定要删除草稿方案"${plan.title}"吗？此操作不可恢复。`,
          '确认删除',
          {
            confirmButtonText: '确定删除',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
        await api.delete(`/api/treatment/plans/${plan.id}/`)
        ElMessage.success('删除成功')
        this.planListKey++
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('删除失败')
        }
      }
    },
    handleViewExecutions(plan) {
      this.selectedPlanId = plan.id
      this.selectedPlan = plan
      this.executionDialogVisible = true
    }
  }
}
</script>

<style scoped>
.doctor-plan-page {
  padding: 20px;
}

.sidebar-card {
  height: calc(100vh - 100px);
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.sidebar-header h3 {
  margin: 0;
  font-size: 16px;
}

.search-input {
  margin-bottom: 15px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
}

.patient-info {
  margin: 5px 0 0 0;
  color: #909399;
  font-size: 14px;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.form-section {
  padding: 10px 0;
}

.form-section h4 {
  margin: 10px 0 15px 0;
  color: #303133;
  font-size: 14px;
  padding-left: 10px;
  border-left: 3px solid #409eff;
}

.unit {
  margin-left: 10px;
  color: #606266;
}

.range-input {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: nowrap;
}

.range-input .el-input {
  flex: 1;
  min-width: 60px;
}

.range-separator {
  color: #909399;
  font-weight: bold;
  padding: 0 2px;
}

.target-group {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
  gap: 10px;
}

.target-label {
  width: 80px;
  font-size: 13px;
  color: #606266;
  flex-shrink: 0;
  text-align: right;
}

.target-group .range-input {
  flex: 1;
}

.target-group .text-input {
  flex: 1;
}

.range-input-min {
  width: 80px !important;
}

.range-input-max {
  width: 80px !important;
}

.unit-select {
  width: 85px !important;
}

:deep(.selected-row) {
  background-color: #ecf5ff;
  cursor: pointer;
}

:deep(.el-divider__text) {
  font-size: 16px;
  font-weight: bold;
}
</style>
