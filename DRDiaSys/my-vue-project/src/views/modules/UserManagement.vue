<template>
  <div class="user-management-container">
    <div class="user-management-box">
      <!-- 头部 -->
      <div class="header">
        <div class="title-container">
          <h2>用户管理</h2>
          <p class="subtitle">管理系统用户账号与权限</p>
        </div>
        <el-button type="primary" class="add-button" @click="handleAdd">
          <el-icon><Plus /></el-icon>
          添加用户
        </el-button>
      </div>

      <!-- 角色统计卡片 -->
      <el-row :gutter="20" class="stat-cards">
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeRole === 'admin' }" @click="filterByRole('admin')">
            <div class="stat-icon admin-icon">
              <el-icon><User /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-name">管理员</div>
              <div class="stat-value">{{ roleStats.admin || 0 }}</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeRole === 'doctor' }" @click="filterByRole('doctor')">
            <div class="stat-icon doctor-icon">
              <el-icon><UserFilled /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-name">医生</div>
              <div class="stat-value">{{ roleStats.doctor || 0 }}</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeRole === 'patient' }" @click="filterByRole('patient')">
            <div class="stat-icon patient-icon">
              <el-icon><Avatar /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-name">患者</div>
              <div class="stat-value">{{ roleStats.patient || 0 }}</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeRole === 'all' }" @click="filterByRole('all')">
            <div class="stat-icon total-icon">
              <el-icon><UserFilled /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-name">用户总数</div>
              <div class="stat-value">{{ roleStats.total || 0 }}</div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 用户列表 -->
      <el-card class="user-list">
        <template #header>
          <div class="card-header">
            <span>用户列表</span>
            <div class="search-container">
              <el-input
                v-model="searchQuery"
                placeholder="搜索用户名/姓名/邮箱"
                class="search-input"
                @keyup.enter="handleSearch"
              >
                <template #prefix>
                  <el-icon><Search /></el-icon>
                </template>
                <template #suffix v-if="searchQuery">
                  <el-icon class="clear-icon" @click="clearSearch"><Close /></el-icon>
                </template>
                <template #append>
                  <el-button @click="handleSearch">搜索</el-button>
                </template>
              </el-input>
              <el-tag v-if="isSearching" type="info" class="search-tag" closable @close="clearSearch">
                搜索结果: {{ total }} 条
              </el-tag>
              <el-tag v-if="activeRole !== 'all'" :type="getRoleType(activeRole)" class="filter-tag" closable @close="filterByRole('all')">
                角色: {{ getRoleLabel(activeRole) }}
              </el-tag>
            </div>
          </div>
        </template>

        <el-table 
          :data="filteredUserList" 
          style="width: 100%"
          :header-cell-style="{
            background: 'rgba(33, 150, 243, 0.1)',
            color: '#333',
            fontWeight: '600'
          }"
          v-loading="tableLoading"
          element-loading-text="加载中..."
        >
          <el-table-column prop="username" label="用户名" />
          <el-table-column prop="name" label="姓名" />
          <el-table-column prop="role" label="角色">
            <template #default="scope">
              <el-tag :type="getRoleType(scope.row.role)" class="role-tag">
                {{ getRoleLabel(scope.row.role) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="createTime" label="创建时间" />
          <el-table-column prop="status" label="状态">
            <template #default="scope">
              <el-tag :type="scope.row.status === 'active' ? 'success' : 'danger'" class="status-tag">
                {{ scope.row.status === 'active' ? '启用' : '禁用' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="250">
            <template #default="scope">
              <el-button
                type="primary"
                link
                @click="handleEdit(scope.row)"
              >
                编辑
              </el-button>
              <el-button
                type="warning"
                link
                @click="handleChangeRole(scope.row)"
              >
                角色
              </el-button>
              <el-button
                type="primary"
                link
                @click="handleResetPassword(scope.row)"
              >
                重置密码
              </el-button>
              <el-button
                :type="scope.row.status === 'active' ? 'danger' : 'success'"
                link
                @click="handleToggleStatus(scope.row)"
              >
                {{ scope.row.status === 'active' ? '禁用' : '启用' }}
              </el-button>
            </template>
          </el-table-column>
        </el-table>

        <div class="pagination">
          <el-pagination
            v-model:current-page="currentPage"
            v-model:page-size="pageSize"
            :total="total"
            :page-sizes="[10, 20, 50, 100]"
            layout="sizes, prev, pager, next"
            @size-change="handleSizeChange"
            @current-change="handleCurrentChange"
          />
        </div>
      </el-card>

      <!-- 添加/编辑用户对话框 -->
      <el-dialog
        v-model="dialogVisible"
        :title="dialogType === 'add' ? '添加用户' : '编辑用户'"
        width="500px"
        class="user-dialog"
      >
        <el-form
          ref="userForm"
          :model="userForm"
          :rules="rules"
          label-position="top"
          class="user-form"
        >
          <el-form-item label="用户名" prop="username">
            <el-input v-model="userForm.username" :disabled="dialogType === 'edit'" />
          </el-form-item>
          <el-form-item label="姓名" prop="name">
            <el-input v-model="userForm.name" />
          </el-form-item>
          <el-form-item label="角色" prop="role">
            <el-select v-model="userForm.role" placeholder="请选择角色" class="role-select">
              <el-option label="管理员" value="admin" />
              <el-option label="医生" value="doctor" />
              <el-option label="患者" value="patient" />
            </el-select>
          </el-form-item>
          <el-form-item label="密码" prop="password" v-if="dialogType === 'add'">
            <el-input v-model="userForm.password" type="password" show-password />
          </el-form-item>
        </el-form>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="dialogVisible = false">取消</el-button>
            <el-button type="primary" @click="submitForm">
              确定
            </el-button>
          </span>
        </template>
      </el-dialog>

      <!-- 角色变更对话框 -->
      <el-dialog
        v-model="roleDialogVisible"
        title="变更用户角色"
        width="400px"
        class="role-dialog"
      >
        <div class="role-form">
          <p class="user-info-text">
            用户名: <b>{{ selectedUser.username }}</b><br>
            当前角色: <el-tag :type="getRoleType(selectedUser.role)" size="small">{{ getRoleLabel(selectedUser.role) }}</el-tag>
          </p>
          <el-form label-position="top">
            <el-form-item label="选择新角色">
              <el-select v-model="newRole" placeholder="请选择新角色" class="role-select">
                <el-option label="管理员" value="admin" />
                <el-option label="医生" value="doctor" />
                <el-option label="患者" value="patient" />
              </el-select>
            </el-form-item>
          </el-form>
        </div>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="roleDialogVisible = false">取消</el-button>
            <el-button type="primary" @click="submitRoleChange">
              确认变更
            </el-button>
          </span>
        </template>
      </el-dialog>
    </div>

    <!-- 装饰效果 -->
    <div class="tech-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
    </div>
  </div>
</template>

<script>
import { Plus, Search, User, UserFilled, Avatar, Close } from '@element-plus/icons-vue'

export default {
  name: 'UserManagement',
  components: {
    Plus,
    Search,
    User,
    UserFilled,
    Avatar,
    Close
  },
  data() {
    return {
      userList: [],
      dialogVisible: false,
      dialogType: 'add',
      searchQuery: '',
      currentPage: 1,
      pageSize: 10,
      total: 0,
      isSearching: false,
      userForm: {
        id: '',
        username: '',
        name: '',
        password: '',
        role: 'patient'
      },
      roleDialogVisible: false,
      selectedUser: {},
      newRole: '',
      roleStats: {
        admin: 0,
        doctor: 0,
        patient: 0,
        total: 0
      },
      rules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
        ],
        name: [
          { required: true, message: '请输入姓名', trigger: 'blur' }
        ],
        role: [
          { required: true, message: '请选择角色', trigger: 'change' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
        ]
      },
      tableLoading: false,
      activeRole: 'all'
    }
  },
  computed: {
    filteredUserList() {
      return this.userList.filter(user =>
        user.username.includes(this.searchQuery) ||
        user.name.includes(this.searchQuery)
      )
    }
  },
  methods: {
    // 辅助函数：规范化角色名称，处理大小写和空格问题
    normalizeRoleName(roleName) {
      if (!roleName) return '';
      // 转为小写并去除前后空格
      return roleName.trim().toLowerCase();
    },
    
    // 获取用户角色统计
    async fetchRoleStatistics() {
      try {
        const response = await this.axios.get('/api/users/role_statistics/')
        
        if (response.status === 200) {
          const data = response.data
          console.log('角色統計API原始數據:', JSON.stringify(data, null, 2));
          
          // 手动获取每个角色的用户数，确保精确匹配
          let adminCount = 0, doctorCount = 0, patientCount = 0;
          
          if (data.role_distribution && Array.isArray(data.role_distribution)) {
            data.role_distribution.forEach(item => {
              const normalizedName = this.normalizeRoleName(item.name);
              const count = parseInt(item.user_count) || 0;
              
              console.log(`处理角色 "${normalizedName}", 用户数: ${count}`);
              
              if (normalizedName === 'admin') {
                adminCount = count;
              } else if (normalizedName === 'doctor') {
                doctorCount = count;
              } else if (normalizedName === 'patient') {
                patientCount = count;
              }
            });
          }
          
          // 创建新对象以触发视图更新
          this.roleStats = {
            admin: adminCount,
            doctor: doctorCount,
            patient: patientCount,
            total: data.total_users || 0
          };
          
          console.log('最终角色统计数据:', this.roleStats);
          
          return Promise.resolve(this.roleStats);
        } else {
          const errorText = JSON.stringify(response.data)
          console.error('獲取角色統計API返回錯誤:', errorText);
          this.updateRoleStats();
          return Promise.reject('獲取角色統計失敗: ' + errorText);
        }
      } catch (error) {
        console.error('獲取角色統計出現異常:', error);
        this.updateRoleStats();
        return Promise.reject(error);
      }
    },
    // 搜索用户
    async handleSearch() {
      console.log(`正在搜索: "${this.searchQuery}"`);
      this.isSearching = !!this.searchQuery;
      // 重置到第一页
      this.currentPage = 1;
      await this.fetchUserList();
    },
    
    // 清除搜索
    async clearSearch() {
      if (this.searchQuery) {
        this.searchQuery = '';
        this.currentPage = 1;
        await this.fetchUserList();
      }
    },
    
    // 获取用户列表
    async fetchUserList() {
      try {
        // 开始加载
        this.tableLoading = true;
        
        // 构建请求参数
        const params = {
          page: this.currentPage,
          page_size: this.pageSize,
          search: this.searchQuery
        };
        
        // 添加角色过滤参数
        if (this.activeRole && this.activeRole !== 'all') {
          params.role = this.activeRole;
        }
        
        console.log(`獲取用戶列表，頁碼: ${this.currentPage}, 每頁數量: ${this.pageSize}${this.searchQuery ? ', 搜索關鍵字: ' + this.searchQuery : ''}${this.activeRole !== 'all' ? ', 角色: ' + this.activeRole : ''}`);
        
        const response = await this.axios.get(`http://localhost:8000/api/users/`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          params: params
        });
        
        if (response.status === 200) {
          // 从响应中获取数据
          const data = response.data;
          console.log('API返回的分页信息:',
            `总用户: ${data.total}, `,
            `当前页: ${data.page}/${data.total_pages}, `,
            `每页: ${data.page_size}条`
          );
          
          this.total = data.total;
          this.isSearching = data.has_search || this.activeRole !== 'all';
          
          // 转换用户数据
          this.userList = data.results.map(user => {
            // 确保组信息正确解析
            let userRole = 'patient'; // 默认角色
            if (user.groups && Array.isArray(user.groups) && user.groups.length > 0) {
              // 取出组名
              userRole = typeof user.groups[0] === 'string' 
                ? user.groups[0] 
                : (user.groups[0].name || 'patient');
            }
            
            // 確保狀態正確解析
            const isActive = user.is_active === true || user.is_active === 'true';
            const status = isActive ? 'active' : 'inactive';
            
            return {
              id: user.id,
              username: user.username,
              name: user.first_name || user.username,
              role: userRole,
              createTime: new Date(user.date_joined).toLocaleString(),
              status: status,
              is_active: isActive,
              groups: user.groups // 保留原始数据以便其他操作使用
            }
          });
          
          if (this.isSearching) {
            const filterType = this.searchQuery ? '搜索' : '角色筛选';
            console.log(`${filterType}结果: 找到 ${this.userList.length} 个匹配用户`);
          }
        }
      } catch (error) {
        console.error('获取用户列表失败:', error);
        this.$message.error(error.response?.data?.message || '獲取用戶列表失敗');
      } finally {
        // 结束加载状态
        this.tableLoading = false;
      }
    },
    // 在API請求失敗時，從本地用戶列表計算統計數據
    updateRoleStats() {
      console.log('正在從本地用戶列表計算角色統計:', this.userList);
      
      let adminCount = 0;
      let doctorCount = 0;
      let patientCount = 0;
      
      // 遍歷用戶列表計算每個角色的用戶數量
      this.userList.forEach(user => {
        // 確保用戶有 groups 屬性，且是一個數組
        if (user.groups && Array.isArray(user.groups)) {
          // 檢查用戶所屬的角色組
          for (const group of user.groups) {
            const normalizedRole = this.normalizeRoleName(group.name || group);
            console.log(`用戶 ${user.username} 的角色: ${normalizedRole}`);
            
            if (normalizedRole === 'admin') {
              adminCount++;
              break; // 一個用戶只計算一次
            } else if (normalizedRole === 'doctor') {
              doctorCount++;
              break;
            } else if (normalizedRole === 'patient') {
              patientCount++;
              break;
            }
          }
        } else {
          console.warn(`用戶 ${user.username} 沒有有效的角色組信息`);
        }
      });
      
      // 輸出計算結果
      console.log('本地計算的角色統計:',
        `管理員: ${adminCount}, 醫生: ${doctorCount}, 患者: ${patientCount}, 總計: ${this.userList.length}`
      );
      
      // 更新統計數據
      this.roleStats = {
        admin: adminCount,
        doctor: doctorCount,
        patient: patientCount,
        total: this.userList.length
      };
    },
    // 添加用戶
    async handleAdd() {
      this.dialogType = 'add'
      this.userForm = {
        username: '',
        name: '',
        role: '',
        password: ''
      }
      this.dialogVisible = true
    },
    // 编辑用户
    handleEdit(user) {
      console.log('開始編輯用戶:', user);
      this.dialogType = 'edit'
      this.userForm = { 
        ...user,
        password: '' // 编辑时不显示密码
      }
      console.log('編輯表單數據:', this.userForm);
      this.dialogVisible = true
    },
    // 重置密码
    async handleResetPassword(user) {
      try {
        await this.$confirm('確認重置該用戶的密碼嗎？', '提示', {
          type: 'warning'
        });
        
        // 用户确认后执行
        try {
          const response = await this.axios.post(`/api/users/${user.id}/reset_password/`, {}, {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          })

          if (response.status === 200) {
            // 获取后端返回的新密码
            const newPassword = response.data.new_password;
            
            // 使用MessageBox显示新密码
            await this.$msgbox({
              title: '密碼重置成功',
              message: `用戶 ${user.username} 的新密碼是：<span style="color: #E6A23C; font-weight: bold; font-size: 18px;">${newPassword}</span><br><br>請記下此密碼並妥善保管！`,
              dangerouslyUseHTMLString: true,
              confirmButtonText: '複製密碼',
              center: true,
              beforeClose: (action, instance, done) => {
                if (action === 'confirm') {
                  // 尝试复制到剪贴板
                  try {
                    navigator.clipboard.writeText(newPassword).then(() => {
                      this.$message.success('密碼已複製到剪貼板');
                      done();
                    }).catch(() => {
                      this.$message.warning('無法自動複製密碼，請手動複製');
                      done();
                    });
                  } catch (e) {
                    this.$message.warning('無法自動複製密碼，請手動複製');
                    done();
                  }
                } else {
                  done();
                }
              }
            });
          }
        } catch (error) {
          this.$message.error(error.response?.data?.message || '密碼重置失敗')
        }
      } catch (err) {
        // 用户取消，不做任何操作
        console.log('用户取消了重置密码操作');
      }
    },
    // 切换用户状态
    async handleToggleStatus(user) {
      // 根據當前狀態確定操作類型
      const action = user.status === 'active' ? '禁用' : '啟用'
      const newIsActive = user.status === 'active' ? false : true
      
      console.log(`準備${action}用戶 ${user.username}，當前狀態：${user.status}，新的is_active值將是：${newIsActive}`);
      
      try {
        await this.$confirm(`確認${action}該用戶嗎？`, '提示', {
          type: 'warning'
        });
        
        // 用户确认后执行
        try {
          console.log(`發送請求到後端，更新用戶 ${user.id} 的 is_active 為 ${newIsActive}`);
          
          const response = await this.axios.put(`http://localhost:8000/api/users/${user.id}/`, {
            is_active: newIsActive
          }, {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          })

          if (response.status === 200) {
            // 更新本地用戶狀態
            user.status = newIsActive ? 'active' : 'inactive'
            user.is_active = newIsActive
            
            console.log(`用戶 ${user.username} 的狀態已更新為：${user.status}`);
            this.$message.success(`${action}成功`)
            
            // 重新獲取用戶列表，確保數據同步
            await this.fetchUserList()
            
            // 更新角色統計數據
            await this.fetchRoleStatistics()
          }
        } catch (error) {
          console.error(`${action}用戶失敗:`, error.response?.data || error);
          this.$message.error(error.response?.data?.message || `${action}失敗`)
        }
      } catch (err) {
        // 用户取消，不做任何操作
        console.log(`用户取消了${action}操作`);
      }
    },
    // 变更角色
    handleChangeRole(user) {
      this.selectedUser = { ...user }
      this.newRole = user.role
      this.roleDialogVisible = true
    },
    // 提交角色变更
    async submitRoleChange() {
      if (this.newRole === this.selectedUser.role) {
        this.$message.info('角色未發生變化')
        this.roleDialogVisible = false
        return
      }
      
      try {
        console.log(`正在將用戶 ${this.selectedUser.username} 的角色從 ${this.selectedUser.role} 變更為 ${this.newRole}`);
        
        const response = await this.axios.put(`http://localhost:8000/api/users/${this.selectedUser.id}/`, {
          groups: [this.newRole]
        }, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })

        if (response.status === 200) {
          this.$message.success('角色變更成功')
          this.roleDialogVisible = false
          
          // 更新本地用戶列表中的角色信息
          const userIndex = this.userList.findIndex(u => u.id === this.selectedUser.id);
          if (userIndex !== -1) {
            this.userList[userIndex].role = this.newRole;
            console.log(`本地用戶列表已更新，用戶 ${this.selectedUser.username} 的角色現在是 ${this.newRole}`);
          }
          
          // 重新獲取統計數據和用戶列表
          await this.fetchRoleStatistics()
          await this.fetchUserList()
        }
      } catch (error) {
        console.error('角色變更失敗:', error.response?.data || error);
        this.$message.error(error.response?.data?.message || '角色變更失敗')
      }
    },
    // 提交表單
    async submitForm() {
      this.$refs.userForm.validate(async (valid) => {
        if (valid) {
          try {
            if (this.dialogType === 'add') {
              console.log('添加新用戶:', this.userForm);
              // 添加用戶
              const response = await this.axios.post('http://localhost:8000/api/users/create/', {
                username: this.userForm.username,
                password: this.userForm.password,
                first_name: this.userForm.name,
                groups: [this.userForm.role] // 設置用戶組
              }, {
                headers: {
                  'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
              })

              if (response.status === 201) {
                this.$message.success('添加用戶成功')
                this.dialogVisible = false
                // 重新獲取統計數據和用戶列表
                await this.fetchRoleStatistics()
                await this.fetchUserList()
              }
            } else {
              console.log('更新用戶:', this.userForm);
              // 更新用戶
              const response = await this.axios.put(`http://localhost:8000/api/users/${this.userForm.id}/`, {
                first_name: this.userForm.name,
                groups: [this.userForm.role]
              }, {
                headers: {
                  'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
              })

              if (response.status === 200) {
                this.$message.success('更新用戶成功')
                this.dialogVisible = false
                
                // 更新本地用戶列表中的數據
                const userIndex = this.userList.findIndex(u => u.id === this.userForm.id);
                if (userIndex !== -1) {
                  this.userList[userIndex].name = this.userForm.name;
                  this.userList[userIndex].role = this.userForm.role;
                  console.log(`本地用戶列表已更新，用戶 ${this.userForm.username} 的角色現在是 ${this.userForm.role}`);
                }
                
                // 重新獲取統計數據和用戶列表
                await this.fetchRoleStatistics()
                await this.fetchUserList()
              }
            }
          } catch (error) {
            console.error('操作用戶失敗:', error.response?.data || error);
            this.$message.error(error.response?.data?.message || '操作失敗')
          }
        }
      })
    },
    // 获取角色标签
    getRoleLabel(role) {
      const roleMap = {
        admin: '管理员',
        doctor: '医生',
        patient: '患者'
      }
      return roleMap[role] || role
    },
    // 获取角色标签类型
    getRoleType(role) {
      const typeMap = {
        admin: 'danger',
        doctor: 'warning',
        patient: 'info'
      }
      return typeMap[role] || ''
    },
    // 分页相关方法
    handleSizeChange(val) {
      console.log(`每页显示数量变更为: ${val}`);
      this.pageSize = val;
      this.currentPage = 1; // 重置到第一页
      this.fetchUserList();
    },
    handleCurrentChange(val) {
      console.log(`页码变更为: ${val}`);
      this.currentPage = val;
      this.fetchUserList();
    },
    // 过滤用户
    async filterByRole(role) {
      if (this.activeRole !== role) {
        console.log(`切换角色筛选: ${this.activeRole} -> ${role}`);
        this.activeRole = role;
        this.currentPage = 1;
        await this.fetchUserList();
      } else if (role === 'all') {
        // 已经是"全部"状态，不需要操作
        console.log('已经是全部用户视图，无需切换');
      }
    }
  },
  created() {
    // 先獲取角色統計，再獲取用戶列表，確保統計數據的獨立性
    this.fetchRoleStatistics().then(() => {
      this.fetchUserList()
    }).catch(() => {
      this.fetchUserList()
    })
  }
}
</script>

<style scoped>
.user-management-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
  position: relative;
  overflow: hidden;
  padding: 10px;
}

.user-management-box {
  width: 100%;
  max-width: 1800px;
  padding: 30px 40px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
  position: relative;
  z-index: 1;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.title-container {
  color: #fff;
}

.title-container h2 {
  font-size: 24px;
  margin: 0;
  font-weight: 600;
}

.subtitle {
  color: #a8b2d1;
  font-size: 14px;
  margin-top: 8px;
}

.add-button {
  background: linear-gradient(45deg, #2196F3, #00BCD4);
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.add-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
}

.user-list {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  border: none;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
}

.card-header span {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.search-container {
  display: flex;
  align-items: center;
}

.search-input {
  width: 250px;
}

.role-tag, .status-tag {
  border-radius: 4px;
  padding: 4px 8px;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
  padding: 0 20px;
}

/* 装饰效果 */
.tech-decoration {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  pointer-events: none;
}

.circle {
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(45deg, rgba(33, 150, 243, 0.1), rgba(0, 188, 212, 0.1));
  animation: float 6s ease-in-out infinite;
}

.circle-1 {
  width: 300px;
  height: 300px;
  top: -150px;
  right: -150px;
  animation-delay: 0s;
}

.circle-2 {
  width: 200px;
  height: 200px;
  bottom: -100px;
  left: -100px;
  animation-delay: 2s;
}

.circle-3 {
  width: 150px;
  height: 150px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  animation-delay: 4s;
}

@keyframes float {
  0% {
    transform: translateY(0) rotate(0deg);
  }
  50% {
    transform: translateY(-20px) rotate(180deg);
  }
  100% {
    transform: translateY(0) rotate(360deg);
  }
}

/* 对话框样式 */
:deep(.user-dialog),
:deep(.role-dialog) {
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
}

:deep(.user-dialog .el-dialog),
:deep(.role-dialog .el-dialog) {
  width: 90% !important;
  max-width: 600px;
  margin: 5vh auto !important;
}

:deep(.el-dialog__header) {
  background: linear-gradient(135deg, #2196F3, #00BCD4);
  margin: 0;
  padding: 22px 24px;
  position: relative;
}

:deep(.el-dialog__title) {
  color: #fff;
  font-weight: 600;
  font-size: 18px;
  letter-spacing: 0.5px;
}

:deep(.el-dialog__headerbtn) {
  top: 20px;
  right: 20px;
}

:deep(.el-dialog__headerbtn .el-dialog__close) {
  color: rgba(255, 255, 255, 0.9);
  font-size: 20px;
  transition: transform 0.3s;
}

:deep(.el-dialog__headerbtn:hover .el-dialog__close) {
  color: #fff;
  transform: rotate(90deg);
}

:deep(.el-dialog__body) {
  padding: 30px 24px;
  background: #f9fafc;
  max-height: 70vh;
  overflow-y: auto;
}

:deep(.el-dialog__footer) {
  padding: 16px 24px 20px;
  border-top: 1px solid #ebeef5;
  background: #f9fafc;
}

:deep(.el-input__inner) {
  border-radius: 10px;
  border: 1px solid #dcdfe6;
  padding: 12px 15px;
  transition: all 0.3s;
  background: #fff;
  height: 50px;
}

:deep(.el-input__wrapper) {
  border-radius: 10px;
  box-shadow: 0 0 0 1px #dcdfe6 inset;
  padding: 1px 15px;
  transition: all 0.3s;
}

:deep(.el-input__wrapper.is-focus),
:deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px #2196F3 inset, 0 5px 10px rgba(33, 150, 243, 0.1);
}

:deep(.el-select .el-input__wrapper) {
  border-radius: 10px;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: #333;
  font-size: 14px;
  padding-bottom: 8px;
}

:deep(.user-form .el-form-item) {
  margin-bottom: 22px;
}

:deep(.dialog-footer .el-button) {
  padding: 12px 24px;
  font-size: 14px;
  border-radius: 10px;
  transition: all 0.3s;
}

:deep(.dialog-footer .el-button--default) {
  border: 1px solid #dcdfe6;
  background: white;
}

:deep(.dialog-footer .el-button--default:hover) {
  border-color: #a0cfff;
  color: #409eff;
  background: rgba(33, 150, 243, 0.05);
  transform: translateY(-2px);
}

:deep(.dialog-footer .el-button--primary) {
  background: linear-gradient(135deg, #2196F3, #00BCD4);
  border: none;
  box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
}

:deep(.dialog-footer .el-button--primary:hover) {
  background: linear-gradient(135deg, #1e88e5, #00acc1);
  transform: translateY(-2px);
  box-shadow: 0 6px 15px rgba(33, 150, 243, 0.4);
}

/* 角色对话框特有样式 */
:deep(.role-dialog .el-dialog__header) {
  background: linear-gradient(135deg, #FF9800, #FFCA28);
}

:deep(.role-dialog .dialog-footer .el-button--primary) {
  background: linear-gradient(135deg, #FF9800, #FFCA28);
  box-shadow: 0 4px 12px rgba(255, 152, 0, 0.3);
}

:deep(.role-dialog .dialog-footer .el-button--primary:hover) {
  background: linear-gradient(135deg, #f57c00, #ffb300);
  box-shadow: 0 6px 15px rgba(255, 152, 0, 0.4);
}

/* 强制用户表单标签顶部对齐 */
:deep(.user-form.el-form--label-top .el-form-item__label) {
  display: block !important;
  text-align: left !important;
  float: none !important;
  width: auto !important;
  line-height: normal !important; /* 确保行高正常 */
  padding-bottom: 8px !important; /* 保留或调整底部填充 */
}

:deep(.user-form.el-form--label-top .el-form-item__content) {
  margin-left: 0 !important;
}

.user-info-text {
  background: rgba(255, 152, 0, 0.05);
  border-left: 4px solid #FF9800;
  padding: 15px;
  border-radius: 0 10px 10px 0;
  margin-bottom: 25px;
  line-height: 1.8;
  color: #333;
}

.user-info-text b {
  color: #333;
  font-weight: 600;
}

:deep(.role-form .el-select) {
  width: 100%;
}

:deep(.role-form .el-form-item__label) {
  font-weight: 500;
  margin-bottom: 10px;
}

:deep(.role-select) {
  width: 100%;
}

:deep(.el-tag) {
  border-radius: 6px;
  padding: 5px 10px;
  font-weight: 500;
}

/* 统计卡片样式 */
.stat-cards {
  margin: 20px 0;
}

.active-card {
  border: 2px solid #409EFF;
  transform: translateY(-5px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
}

.active-card .stat-icon {
  transform: scale(1.1);
}

.stat-card {
  cursor: pointer;
  height: 100px;
  display: flex;
  align-items: center;
  padding: 20px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.9);
  transition: all 0.3s;
  margin-bottom: 15px;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

.stat-icon {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-right: 15px;
}

.stat-icon .el-icon {
  font-size: 24px;
  color: white;
}

.admin-icon {
  background: linear-gradient(45deg, #f44336, #ff7043);
}

.doctor-icon {
  background: linear-gradient(45deg, #ff9800, #ffca28);
}

.patient-icon {
  background: linear-gradient(45deg, #2196f3, #00bcd4);
}

.total-icon {
  background: linear-gradient(45deg, #4caf50, #8bc34a);
}

.stat-info {
  flex: 1;
}

.stat-name {
  font-size: 14px;
  color: #666;
  margin-bottom: 5px;
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: #333;
}

.search-tag {
  margin-left: 10px;
  border-radius: 4px;
  padding: 0 8px;
  height: 32px;
  line-height: 30px;
  font-size: 12px;
}

.filter-tag {
  margin-left: 10px;
  border-radius: 4px;
  padding: 0 8px;
  height: 32px;
  line-height: 30px;
  font-size: 12px;
}
</style> 