<template>
  <!-- 注意：為了確保 el-container 佔滿高度，
              需要全局重置 html, body, #app 的 margin, padding, height -->
  <el-container class="layout-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapse ? '64px' : '180px'">
      <div class="logo">
        <img src="../assets/logo.svg" alt="Logo" class="logo-img">
        <span v-show="!isCollapse">DR诊断系统</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        class="el-menu-vertical"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
        :collapse="isCollapse"
        router
      >
        <template v-for="item in menuItems" :key="item.path || item.title">
          <!-- 有子菜单的项 -->
          <el-sub-menu v-if="item.children && item.children.length > 0" :index="item.path || item.title">
            <template #title>
              <el-icon><component :is="item.icon" /></el-icon>
              <span>{{ item.label }}</span>
            </template>
        <el-menu-item
              v-for="child in item.children"
              :key="child.path"
              :index="child.path"
            >
              <span>{{ child.label }}</span>
            </el-menu-item>
          </el-sub-menu>
          <!-- 普通菜单项 -->
          <el-menu-item v-else :index="item.path">
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </el-menu-item>
        </template>
      </el-menu>
    </el-aside>

    <!-- 主要内容区 -->
    <el-container class="main-content-container">
      <!-- 顶部导航栏 -->
      <el-header>
        <div class="header-left">
          <el-icon class="toggle-sidebar" @click="toggleSidebar">
            <Fold v-if="!isCollapse" />
            <Expand v-else />
          </el-icon>
          <el-breadcrumb separator="/">
            <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
            <el-breadcrumb-item v-if="currentRoute">{{ currentRoute }}</el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="header-right">
          <el-dropdown>
            <span class="user-info">
              <el-avatar :size="32" :src="userAvatar" />
              <span>{{ userName }}</span>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>个人信息</el-dropdown-item>
                <el-dropdown-item>修改密码</el-dropdown-item>
                <el-dropdown-item divided @click="logout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>
      <!-- 内容区 -->
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script>
import {
  DataLine,
  Picture,
  Search,
  Document,
  Files,
  User,
  Fold,
  Expand,
  Edit,
  View,
  ChatDotRound,
  Notebook,
  Promotion,
  UserFilled
} from '@element-plus/icons-vue'

export default {
  name: 'MainLayout',
  components: {
    DataLine,
    Picture,
    Search,
    Document,
    Files,
    User,
    Fold,
    Expand,
    Edit,
    View,
    ChatDotRound,
    Notebook,
    Promotion,
    UserFilled
  },
  data() {
    return {
      isCollapse: false,
      userRole: localStorage.getItem('userRole') || 'admin',
      userName: localStorage.getItem('userName') || '管理员',
      userAvatar: 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png '
    }
  },
  computed: {
    activeMenu() {
      const path = this.$route.path
      if (path === '/') {
        return '/dashboard'
      }
      // 如果是子菜单路径，返回子菜单路径用于高亮
      return path
    },
    currentRoute() {
      const routeMap = {
        '/dashboard': '首页',
        '/image-processing': '图像管理与预处理',
        '/disease-detection': '病变检测',
        '/user-management': '用户管理',
        // 用户角色路由
        '/personal-info': '个人信息录入',
        '/condition-info': '病情信息录入',
        '/eye-image-view': '眼部影像查看',
        '/report-view': '报告复核',
        '/doctor-patient-chat': '医患交流',
        // 医生角色路由
        '/medical-record': '病历管理',
        '/treatment-plan': '方案推荐'
      }
      return routeMap[this.$route.path] || '';
    },
    // 根据角色过滤菜单项
    menuItems() {
      // 用户（patient）菜单
      const patientMenus = [
        {
          title: 'info-entry',
          label: '信息录入',
          icon: 'Edit',
          children: [
            { path: '/personal-info', label: '个人信息录入' },
            { path: '/condition-info', label: '病情信息录入' }
          ]
        },
        { path: '/eye-image-view', icon: 'View', label: '眼部影像查看' },
        { path: '/report-view', icon: 'Document', label: '诊断报告' },
        { path: '/doctor-patient-chat', icon: 'ChatDotRound', label: '医患交流' }
      ]

      // 医生（doctor）菜单
      const doctorMenus = [
        { path: '/report-view', icon: 'Document', label: '报告复核' },
        { path: '/medical-record', icon: 'Notebook', label: '病历管理' },
        { path: '/treatment-plan', icon: 'Promotion', label: '方案推荐' },
        { path: '/doctor-patient-chat', icon: 'ChatDotRound', label: '医患交流' }
      ]

      // 管理员（admin）菜单
      const adminMenus = [
        { path: '/dashboard', icon: 'DataLine', label: '首页' },
        { path: '/image-processing', icon: 'Picture', label: '图像管理与预处理' },
        { path: '/disease-detection', icon: 'Search', label: '病变检测' },
        { path: '/report-view', icon: 'Files', label: 'AI诊断报告' },
        { path: '/user-management', icon: 'UserFilled', label: '用户管理' }
      ]

      // 根据角色返回对应菜单
      if (this.userRole === 'patient') {
        return patientMenus
      } else if (this.userRole === 'doctor') {
        return doctorMenus
      } else {
        // 默认返回管理员菜单
        return adminMenus
      }
    }
  },
  methods: {
    toggleSidebar() {
      this.isCollapse = !this.isCollapse
    },

    // 退出登录
    logout() {
      // 清除本地存储
      localStorage.removeItem('token')
      localStorage.removeItem('userRole')
      localStorage.removeItem('userName')
      localStorage.removeItem('refresh')
      // 使用 window.location 强制刷新页面，确保登录页正确显示
      window.location.href = '/login'
    }
  },
  created() {
    const token = localStorage.getItem('token')
    if (!token && this.$route.path !== '/login') {
      this.$router.replace('/login')
    }
    this.userName = localStorage.getItem('userName') || '管理员';
    this.userRole = localStorage.getItem('userRole') || 'admin';
  }
}
</script>

<style scoped>
.layout-container {
  height: 100vh;
  display: flex;
  flex-direction: row;
}

.el-aside {
  background-color: #304156;
  transition: width 0.3s;
  overflow-x: hidden;
  flex-shrink: 0;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  color: #fff;
  font-size: 18px;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.logo-img {
  width: 32px;
  height: 32px;
  margin-right: 12px;
  flex-shrink: 0;
}



.el-menu-vertical {
    border-right: none;
    height: calc(100vh - 60px);
    overflow-y: auto;
}


.main-content-container {
    flex-direction: column;
    flex-grow: 1;
    overflow: hidden;
}


.el-header {
  background-color: #fff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 20px;
}

.toggle-sidebar {
  font-size: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.header-right {
  display: flex;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 0 10px;
}

.el-main {
  background-color: #f0f2f5;
  padding: 0;
  flex-grow: 1;
  overflow: auto;
}

/* Element Plus 菜單項內部樣式，確保圖標和文字對齊 */
:deep(.el-menu-item) {
  display: flex;
  align-items: center;
  white-space: nowrap;
}

:deep(.el-menu-item > .el-icon) {
  margin-right: 10px;
  flex-shrink: 0;
}

:deep(.el-menu-item span) {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 子菜单样式 */
:deep(.el-sub-menu__title) {
  display: flex;
  align-items: center;
  white-space: nowrap;
}

:deep(.el-sub-menu__title > .el-icon) {
  margin-right: 10px;
  flex-shrink: 0;
}

:deep(.el-sub-menu__title span) {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
}

:deep(.el-sub-menu .el-menu-item) {
  padding-left: 50px !important;
}

.el-menu--collapse  .el-menu-vertical .el-tooltip__trigger {
    padding: 0 20px;
}
</style>
